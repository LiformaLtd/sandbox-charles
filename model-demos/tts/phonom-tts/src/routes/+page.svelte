<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { invalidateAll } from '$app/navigation';
	import type { PageData } from './$types';
	import { models, languages, modelSource, sampleText, type ModelKey, type Language } from '../lib/models';
	import { runLocalPhonon } from '../lib/phonon-server';
	import { OnnxClient } from '../lib/onnx-client';
	import { onnxVersions, onnxVersion, type RuntimeMode, type OnnxDevice } from '../lib/onnx-options';
	import type { PhononTTS } from 'phonon-tts';
	import catalogue from '../lib/voices-catalogue.json';
	import VoiceBlender from '../lib/VoiceBlender.svelte';
	import { blendVoiceKey, blendVoiceUrl, averagedVoices, averageVoiceUrl, type SavedBlend } from '../lib/voice-blends';
	let { data }: { data: PageData } = $props();
	let mounted = $state(false);
	onMount(() => { mounted = true; });
	import workerUrl from '../lib/phonon.worker.ts?worker&url';
	import wasmUrl from 'phonon-tts/wasm/phonon_tts_bg.wasm?url';
	import simdWasmUrl from 'phonon-tts/wasm/phonon_tts_simd_bg.wasm?url';

	const englishPrompts = {
		'Pocket TTS comparison': 'The rain had stopped by morning. Across the river, a small café was already open, and the smell of fresh bread drifted down the street.',
		'Numbers and units': 'Order number 3047 arrives on Thursday. The parcel weighs 2.75 kilograms, and delivery costs £12.50.',
		'Clear consonants': 'She sells seashells by the seashore. Six slick swans swam swiftly past the stone steps.'
	};
	const voiceBase = 'https://huggingface.co/gradium/phonon-voices/resolve/0ee17205f254b5abc43b266c8a0f215565c352f7/';
	const localFiles = import.meta.glob<string>('../lib/local-voices/*.safetensors', {
		eager: true, query: '?url', import: 'default'
	});
	const localVoices = Object.entries(localFiles).map(([path, url]) => ({
		name: path.split('/').pop()!.replace(/\.safetensors$/, ''),
		url
	}));
	const voiceFiles = new Map(
		Object.values(catalogue).flat().map(({ name, file }) => [name, voiceBase + file])
	);
	for (const { name, url } of localVoices) voiceFiles.set(name, url);
	for (const group of averagedVoices) voiceFiles.set(group.name, averageVoiceUrl(group.id));
	let text = $state(englishPrompts['Pocket TTS comparison']);
	let modelKey = $state<ModelKey>('31m-en');
	let language = $state<Language>('en');
	const selectedModel = $derived(models[modelKey]);
	const available = $derived(data.modelAvailability[modelKey]);
	const voiceGroups = $derived(Object.fromEntries(Object.entries(catalogue).map(([label, presets]) =>
		[label, presets.filter(v => v.language === language)]).filter(([, presets]) => presets.length)) as typeof catalogue);
	const prompts = $derived(language === 'en' ? englishPrompts : { 'Sample paragraph': sampleText[language] });
	let voice = $state('Easton');
	let mode = $state<RuntimeMode>('browser');
	let onnxDevice = $state<OnnxDevice>('wasm');
	let q8Device = $state<'cpu' | 'webgpu'>('cpu');
	const onnx = new OnnxClient();
	const runtimeLabels: Record<string, string> = { browser: 'Browser Q8', native: 'Local native · 31M English · CPU/Q8', mlx: 'Local MLX · 31M English · Apple GPU' };
	let recordedRuntime = $state('');
	let recordedMode = $state('browser');
	let busy = $state(false);
	let blendPreparing = $state(false);
	let savedBlends = $state<SavedBlend[]>([]);
	let status = $state('Ready');
	let failure = $state('');
	let first = $state<number>();
	let setup = $state<number>();
	let generation = $state<number>();
	let duration = $state<number>();
	let wav = $state('');
	let recordedVoice = $state('');
	let replay = $state<HTMLAudioElement>();
	let context: AudioContext | undefined;
	let controller: AbortController | undefined;
	let engine: PhononTTS | undefined;
	let engineKey = '';
	let recordedModel = $state('31m-en');
	const addedVoices = new Set<string>();
	let destroyed = false;

	function updateBlends(voices: SavedBlend[], selectedId?: string) {
		for (const previous of savedBlends) voiceFiles.delete(blendVoiceKey(previous.id));
		savedBlends = voices;
		for (const item of voices) voiceFiles.set(blendVoiceKey(item.id), blendVoiceUrl(item.id));
		if (selectedId) { voice = blendVoiceKey(selectedId); mode = 'browser'; }
		else if (voice.startsWith('blend-') && !voiceFiles.has(voice)) voice = Object.values(catalogue).flat().find(v => v.language === language)!.name;
	}

	function chooseLanguage(next: Language) {
		language = next;
		const matching = Object.values(catalogue).flat().filter(v => v.language === language);
		if (!averagedVoices.some(v => v.name === voice) && !savedBlends.some(v => blendVoiceKey(v.id) === voice) && !matching.some(v => v.name === voice)) voice = matching[0]?.name || 'Easton';
		if (Object.values(sampleText).includes(text) || Object.values(englishPrompts).includes(text)) text = sampleText[language];
	}

	function chooseModel(next: ModelKey) {
		modelKey = next;
		if (!models[next].languages.includes(language)) chooseLanguage(models[next].languages[0] as Language);
		if (next !== '31m-en') mode = 'browser';
	}

	function stop() {
		controller?.abort();
		const audio = context;
		context = undefined;
		if (audio) void audio.close().catch(() => {});
	}

	onDestroy(() => {
		destroyed = true;
		stop();
		engine?.dispose();
		onnx.dispose();
		if (wav) URL.revokeObjectURL(wav);
	});

	async function generate(preview?: { id: string; name: string; source: Blob }) {
		if (busy || !text.trim() || !available) return;
		stop();
		replay?.pause();
		if (wav) URL.revokeObjectURL(wav);
		wav = '';
		failure = '';
		first = setup = generation = duration = undefined;
		busy = true;
		const run = new AbortController();
		controller = run;
		const { signal } = run;
		const started = performance.now();
		const version = onnxVersion(mode);
		recordedRuntime = version ? `Browser ONNX · ${onnxVersions[version].name} · ${onnxDevice === 'wasm' ? 'CPU' : 'WebGPU'}` : runtimeLabels[mode];
		recordedMode = version ? `${mode}-${onnxDevice}` : mode === 'browser' ? `browser-q8-${q8Device}` : mode;
		if (mode === 'browser') recordedRuntime += ` · ${q8Device === 'cpu' ? 'CPU' : 'WebGPU'}`;
		recordedModel = modelKey;
		recordedRuntime += ` · ${selectedModel.name}${selectedModel.languages.length > 1 ? ` · ${languages[language]}` : ''}`;
		try {
			if (modelKey !== '31m-en' && mode !== 'browser') throw new Error('40M models currently use the Browser Q8 runtime.');
			const audio = new AudioContext({ latencyHint: 'interactive' });
			context = audio;
			await audio.resume();
			signal.throwIfAborted();
			if (version) {
				// One ONNX variant in memory at a time; the worker is reused for warm runs.
				if (engine) { engine.dispose(); engine = undefined; addedVoices.clear(); }
				const chosenVoice = voice;
				const preset = Object.values(catalogue).flat().find(v => v.name === chosenVoice);
				if (!preset) throw new Error('Choose an English preset voice for ONNX.');
				status = `Loading ${recordedRuntime}…`;
				const result = await onnx.run({ version, device: onnxDevice, text: text.trim(), voiceFile: preset.file.replace('.safetensors', '.bin') }, audio, signal,
					message => { status = message; },
					ms => { setup = ms; status = 'Generating speech…'; },
					() => { first = performance.now() - started; status = 'Playing speech…'; });
				signal.throwIfAborted();
				setup = result.setup; generation = result.generation; duration = result.duration;
				wav = URL.createObjectURL(result.audio); recordedVoice = chosenVoice;
				status = 'Generation complete';
				return;
			}
			onnx.dispose();
			if (mode === 'native' || mode === 'mlx') {
				status = `Loading ${runtimeLabels[mode]}…`;
				const chosenVoice = voice;
				const result = await runLocalPhonon({ backend: mode, text: text.trim(), voice: chosenVoice }, audio, signal,
					ms => { setup = ms; status = 'Generating speech…'; },
					() => { first = performance.now() - started; status = 'Streaming speech…'; });
				signal.throwIfAborted();
				setup = result.setup; generation = result.generation; duration = result.duration;
				wav = URL.createObjectURL(result.audio); recordedVoice = chosenVoice;
				status = 'Generation complete';
				return;
			}
			const { PhononTTS, concatPcm, encodeWav } = await import('phonon-tts');
			signal.throwIfAborted();
			const desiredEngine = `${modelKey}/${language}/${q8Device}`;
			if (engine && engineKey !== desiredEngine) { engine.dispose(); engine = undefined; addedVoices.clear(); }
			if (!engine) {
				status = `Loading Phonon ${selectedModel.name}…`;
				const loaded = await PhononTTS.load({
					lang: language, model: modelSource(modelKey), quant: 'q8', device: q8Device, threads: 1,
					workerUrl, wasmUrl, simdWasmUrl,
					onProgress: ({ file, loaded, total, cached }) => {
						if (!signal.aborted && !cached) {
							const amount = total ? ` · ${Math.min(100, Math.round(loaded / total * 100))}%` : '';
							status = `Loading ${file.split('/').pop()}${amount}`;
						}
					}
				});
				if (destroyed) {
					loaded.dispose();
					return;
				}
				engine = loaded; engineKey = desiredEngine;
			}
			signal.throwIfAborted();
			const chosenVoice = preview?.id || voice;
			if (!addedVoices.has(chosenVoice)) {
				status = `Loading ${chosenVoice}…`;
				const file = preview?.source || voiceFiles.get(chosenVoice);
				if (!file) throw new Error('Choose an available voice.');
				await engine.addVoice(chosenVoice, file);
				addedVoices.add(chosenVoice);
			}
			signal.throwIfAborted();
			setup = performance.now() - started;
			status = 'Generating speech…';
			const synthesisStarted = performance.now();
			const stream = engine.stream(text.trim(), { voice: chosenVoice, signal });
			const chunks: Float32Array[] = [];
			let next = audio.currentTime;
			try {
				for await (const pcm of stream) {
					if (signal.aborted) break;
					if (first === undefined) {
						first = performance.now() - started;
						status = 'Streaming speech…';
					}
					const buffer = audio.createBuffer(1, pcm.length, engine.sampleRate);
					buffer.getChannelData(0).set(pcm);
					const source = audio.createBufferSource();
					source.buffer = buffer;
					source.connect(audio.destination);
					next = Math.max(next, audio.currentTime);
					source.start(next);
					next += buffer.duration;
					chunks.push(pcm);
				}
			} finally {
				// Wait for the worker to finish before allowing another request.
				if (signal.aborted) await stream.done.catch(() => {});
			}
			const stats = await stream.done;
			signal.throwIfAborted();
			if (!chunks.length) throw new Error('Phonon returned no audio.');
			generation = stats.totalMs ?? performance.now() - synthesisStarted;
			const pcm = concatPcm(chunks);
			duration = pcm.length / engine.sampleRate;
			wav = URL.createObjectURL(encodeWav(pcm, engine.sampleRate));
			recordedVoice = preview?.name || savedBlends.find(v => blendVoiceKey(v.id) === chosenVoice)?.name || chosenVoice;
			status = 'Generation complete';
		} catch (error) {
			if (signal.aborted) {
				status = 'Stopped';
			} else {
				console.error('Phonon generation failed', error);
				failure = error instanceof Error ? error.message : String(error);
				status = 'Generation failed';
				stop();
			}
		} finally {
			busy = false;
			controller = undefined;
		}
	}
</script>

<svelte:head>
	<title>Phonon TTS Lab</title>
	<meta name="description" content="Compare Phonon English, French, German, Spanish, Portuguese and multilingual speech models." />
</svelte:head>

<main>
	<header>
		<p class="eyebrow">PHONON · FIVE LANGUAGES</p>
		<h1>Phonon TTS Lab</h1>
		<p>Compare Phonon 31M English, the five language-specific 40M models, and 40M multilingual. Choose a model, language, voice and runtime.</p>
	</header>

	<form onsubmit={(event) => { event.preventDefault(); void generate(); }}>
		<fieldset disabled={blendPreparing}>
		<div class="controls model-controls">
			<label for="model">Model</label>
			<select id="model" value={modelKey} disabled={busy || !mounted} onchange={e => chooseModel(e.currentTarget.value as ModelKey)}>
				{#each Object.entries(models) as [key, model] (key)}
					<option value={key}>{model.name} · {(model.bytes / 1e6).toFixed(1)} MB Q8{data.modelAvailability[key as ModelKey] ? '' : ' · needs download'}</option>
				{/each}
			</select>
			<label for="language">Speech language</label>
			<select id="language" value={language} disabled={busy || !mounted || selectedModel.languages.length === 1} onchange={e => chooseLanguage(e.currentTarget.value as Language)}>
				{#each selectedModel.languages as lang (lang)}
					<option value={lang}>{languages[lang as Language]}</option>
				{/each}
			</select>
		</div>
		<p class="hint">Gradium · <a href={`https://huggingface.co/${selectedModel.repo}`} target="_blank" rel="noreferrer">{selectedModel.license}</a>{selectedModel.local ? ' · Non-commercial licence' : ' · Commercial use permitted under its terms'}</p>
		{#if !available}
			<p class="hint" role="status">This model needs a local download before it can run. <a href={`https://huggingface.co/${selectedModel.repo}`} target="_blank" rel="noreferrer">Enable model access on Hugging Face</a>, then download its files locally.</p>
			<button type="button" class="secondary" disabled={busy || !mounted} onclick={() => { void invalidateAll(); }}>Refresh model availability</button>
		{/if}
		{#if modelKey !== '31m-en'}<p class="hint">40M models use Browser Q8 on CPU or WebGPU. Published ONNX exports are available for 31M English only.</p>{/if}
		<div class="row speech-row">
			<label for="speech">Text to speak</label>
			<span>{text.length} / 1000</span>
		</div>
		<textarea id="speech" bind:value={text} rows="4" maxlength="1000" required disabled={busy || !mounted}></textarea>
		<div class="examples" aria-label="Example prompts">
			{#each Object.entries(prompts) as [name, example] (name)}
				<button type="button" class="secondary" disabled={busy || !mounted} onclick={() => { text = example; }}>{name}</button>
			{/each}
		</div>
		<div class="controls">
   <label for="runtime">Run on</label>
   <select id="runtime" value={mode} disabled={busy || !mounted} onchange={e => {
    mode = e.currentTarget.value as typeof mode;
    if (mode !== 'browser' && !Object.values(catalogue).flat().some(v => v.name === voice && v.language === 'en')) voice = 'Easton';
   }}>
    <optgroup label="Original runtimes">
     <option value="browser">Browser Q8 · CPU or WebGPU</option>
     <option value="native" disabled={modelKey !== '31m-en'}>Local native · 31M English · CPU/Q8</option>
     <option value="mlx" disabled={modelKey !== '31m-en'}>Local MLX · 31M English · Apple GPU</option>
    </optgroup>
    <optgroup label="Browser ONNX · Phonon 31M English" disabled={modelKey !== '31m-en'}>
     {#each Object.entries(onnxVersions) as [version, option] (version)}
      <option value={`onnx-${version}`}>{option.name} · {option.size}</option>
     {/each}
    </optgroup>
   </select>
   {#if mode === 'browser'}
    <label for="q8-device">Run Q8 on</label>
    <select id="q8-device" bind:value={q8Device} disabled={busy || !mounted}>
     <option value="cpu">Browser CPU (WASM)</option>
     <option value="webgpu">Browser GPU (WebGPU)</option>
    </select>
   {/if}
   {#if onnxVersion(mode)}
    <label for="onnx-device">Run ONNX on</label>
    <select id="onnx-device" bind:value={onnxDevice} disabled={busy || !mounted}>
     <option value="wasm">Browser CPU (WASM)</option>
     <option value="webgpu">Browser GPU (WebGPU)</option>
    </select>
   {/if}
			<label for="voice">Voice</label>
			<select id="voice" bind:value={voice} disabled={busy || !mounted}>
				{#each Object.entries(voiceGroups) as [label, presets] (label)}
					<optgroup {label}>
						{#each presets as { name } (name)}<option value={name}>{name}</option>{/each}
					</optgroup>
				{/each}
				{#if mode === 'browser' && localVoices.length}
					<optgroup label="Local voices">
						{#each localVoices as { name } (name)}<option value={name}>{name}</option>{/each}
					</optgroup>
				{/if}
				{#if mode === 'browser' && savedBlends.length}
					<optgroup label="Saved blends">{#each savedBlends as item (item.id)}<option value={blendVoiceKey(item.id)}>{item.name}</option>{/each}</optgroup>
				{/if}
				{#if mode === 'browser'}
					<optgroup label="Averaged English voices">{#each averagedVoices as group (group.id)}<option value={group.name}>{group.name} · {group.members.length} voices</option>{/each}</optgroup>
				{/if}
			</select>
			<div class="buttons">
				<button type="submit" disabled={busy || !mounted || !text.trim() || !available}>Generate</button>
				<button type="button" class="secondary" disabled={!busy && !wav} onclick={() => { stop(); replay?.pause(); }}>Stop playback</button>
			</div>
		</div>
		<p class="hint">{onnxVersion(mode) ? 'Downloads the selected ONNX version once and caches it in this browser. Sizes include both graphs and weights; voices are extra. FP16 on WebGPU requires shader-f16 support.' : mode === 'browser' ? 'The first browser run downloads the selected model. Later runs reuse the browser cache. Changing model or language prepares a new runtime.' : 'Uses the Phonon service on port 8184. Models and all 37 voices are stored locally.'}</p>
		</fieldset>
	</form>

	<VoiceBlender disabled={busy || !mounted} canPreview={available && Boolean(text.trim())}
		onworking={working => { blendPreparing = working; }} onvoiceschange={updateBlends}
		onpreview={async preview => { mode = 'browser'; await generate(preview); }} />

	<section aria-labelledby="result">
		<div class="row"><h2 id="result">Streaming result</h2><span role="status">{status}</span></div>
		{#if recordedRuntime}<p class="hint">{recordedRuntime}{recordedVoice && wav ? ` · ${recordedVoice}` : ''}</p>{/if}
		{#if failure}<p class="error" role="alert">{failure}</p>{/if}
		<dl>
			<div><dt>Time to first audio</dt><dd>{first === undefined ? '—' : `${Math.round(first)} ms`}</dd></div>
			<div><dt>Total generation</dt><dd>{generation === undefined ? '—' : `${Math.round(generation)} ms`}</dd></div>
			<div><dt>Setup</dt><dd>{setup === undefined ? '—' : `${Math.round(setup)} ms`}</dd></div>
			<div><dt>Audio duration</dt><dd>{duration === undefined ? '—' : `${duration.toFixed(2)} s`}</dd></div>
			<div><dt>Generation speed</dt><dd>{duration && generation ? `${(duration * 1000 / generation).toFixed(1)}× real time` : '—'}</dd></div>
		</dl>
		{#if wav}
			<audio bind:this={replay} controls src={wav} aria-label="Replay generated speech" onplay={stop}></audio>
			<a class="download" href={wav} download={`phonon-${recordedModel}-${recordedMode}-${recordedVoice.toLowerCase()}.wav`}>Download WAV ↓</a>
		{/if}
		<p class="hint">Time to first audio includes setup. Total generation measures synthesis after setup. ONNX streams blocks of up to 640 ms of audio; the model stays loaded between runs.</p>
	</section>
</main>

<style>
	:global(*) { box-sizing: border-box; }
	:global(body) { margin: 0; background: #121524; color: #f1f1ff; color-scheme: dark; font: 16px/1.5 system-ui, sans-serif; }
	main { max-width: 880px; margin: auto; padding: clamp(24px, 7vw, 78px) clamp(16px, 4vw, 40px); }
	header { margin-bottom: 28px; }
	h1 { margin: 6px 0; font-size: clamp(2.5rem, 8vw, 5rem); letter-spacing: -.06em; line-height: 1.1; }
	header p { margin: 0; color: #bcbfda; }
	header .eyebrow { color: #bfb0ff; font: .8rem ui-monospace, monospace; letter-spacing: .13em; }
	form, section { border: 1px solid #3c4061; background: #1b2035; border-radius: 16px; padding: clamp(14px, 4vw, 24px); }
	fieldset { border: 0; padding: 0; margin: 0; min-width: 0; }
	section { margin-top: 16px; }
	.row, .controls, .buttons, .examples { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
	.row { justify-content: space-between; margin-bottom: 8px; }
	.row span, .hint { color: #bcbfda; font-size: .82rem; }
	textarea, select, button { font: inherit; border-radius: 8px; padding: 8px 12px; }
	textarea, select { border: 1px solid #5a6084; background: #121524; color: inherit; }
	select { min-width: 0; max-width: 100%; }
	textarea { width: 100%; resize: vertical; display: block; }
	.examples, .controls { margin-top: 12px; }
	.examples button { font-size: .8rem; }
	.buttons { margin-left: auto; }
	button { border: 1px solid #c5b6ff; background: #c5b6ff; color: #1b1437; font-weight: 650; cursor: pointer; }
	button.secondary { border-color: #67608e; background: transparent; color: #ded3ff; }
	button:disabled { opacity: .5; cursor: default; }
	h2 { margin: 0; font-size: 1.1rem; }
	dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; }
	dl div { padding: 10px; background: #121524; border-radius: 8px; }
	dt { color: #bcbfda; font-size: .8rem; }
	dd { margin: 4px 0 0; font-size: 1.05rem; font-variant-numeric: tabular-nums; }
	audio { width: 100%; }
	.download { color: #ded3ff; }
	.hint { margin-bottom: 0; }
	.hint a { color: #ded3ff; }
	.model-controls { margin-top: 0; margin-bottom: 14px; }
	.speech-row { margin-top: 14px; }
	.error { color: #ffb4ad; overflow-wrap: anywhere; }
</style>

<script lang="ts">
	import { onDestroy } from 'svelte';
	import { runLocalPhonon } from '../lib/phonon-server';
	import type { PhononTTS } from 'phonon-tts';
	import catalogue from '../lib/english-voices.json';
	import workerUrl from '../lib/phonon.worker.ts?worker&url';
	import wasmUrl from 'phonon-tts/wasm/phonon_tts_bg.wasm?url';
	import simdWasmUrl from 'phonon-tts/wasm/phonon_tts_simd_bg.wasm?url';

	const prompts = {
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
	let text = $state(prompts['Pocket TTS comparison']);
	let voice = $state('Easton');
	let mode = $state<'browser' | 'native' | 'mlx'>('browser');
	const runtimeLabels = { browser: 'Browser WASM · CPU/Q8', native: 'Local native · CPU/Q8', mlx: 'Local MLX · Apple GPU' };
	let recordedRuntime = $state('');
	let recordedMode = $state('browser');
	let busy = $state(false);
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
	const addedVoices = new Set<string>();
	let destroyed = false;

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
		if (wav) URL.revokeObjectURL(wav);
	});

	async function generate() {
		if (busy || !text.trim()) return;
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
		recordedRuntime = runtimeLabels[mode]; recordedMode = mode;
		try {
			const audio = new AudioContext({ latencyHint: 'interactive' });
			context = audio;
			await audio.resume();
			signal.throwIfAborted();
			if (mode !== 'browser') {
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
			if (!engine) {
				status = 'Loading Phonon 31M…';
				const loaded = await PhononTTS.load({
					lang: 'en', quant: 'q8', device: 'cpu', threads: 1,
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
				engine = loaded;
			}
			signal.throwIfAborted();
			const chosenVoice = voice;
			if (!addedVoices.has(chosenVoice)) {
				status = `Loading ${chosenVoice}…`;
				const file = voiceFiles.get(chosenVoice);
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
			recordedVoice = chosenVoice;
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
	<meta name="description" content="Test Phonon 31M English streaming speech in your browser." />
</svelte:head>

<main>
	<header>
		<p class="eyebrow">PHONON 31M · ENGLISH · THREE RUNTIMES</p>
		<h1>Phonon TTS Lab</h1>
		<p>Compare Phonon 31M in your browser, on the native CPU runtime, and on Apple’s GPU with MLX.</p>
	</header>

	<form onsubmit={(event) => { event.preventDefault(); void generate(); }}>
		<div class="row">
			<label for="speech">Text to speak</label>
			<span>{text.length} / 1000</span>
		</div>
		<textarea id="speech" bind:value={text} rows="4" maxlength="1000" required disabled={busy}></textarea>
		<div class="examples" aria-label="Example prompts">
			{#each Object.entries(prompts) as [name, example] (name)}
				<button type="button" class="secondary" disabled={busy} onclick={() => { text = example; }}>{name}</button>
			{/each}
		</div>
		<div class="controls">
   <label for="runtime">Run on</label>
   <select id="runtime" value={mode} disabled={busy} onchange={e => {
    mode = e.currentTarget.value as typeof mode;
    if (mode !== 'browser' && !Object.values(catalogue).flat().some(v => v.name === voice)) voice = 'Easton';
   }}>
    <option value="browser">Browser WASM · CPU/Q8</option>
    <option value="native">Local native · CPU/Q8</option>
    <option value="mlx">Local MLX · Apple GPU</option>
   </select>
			<label for="voice">Voice</label>
			<select id="voice" bind:value={voice} disabled={busy}>
				{#each Object.entries(catalogue) as [label, presets] (label)}
					<optgroup {label}>
						{#each presets as { name } (name)}<option value={name}>{name}</option>{/each}
					</optgroup>
				{/each}
				{#if mode === 'browser' && localVoices.length}
					<optgroup label="Local voices">
						{#each localVoices as { name } (name)}<option value={name}>{name}</option>{/each}
					</optgroup>
				{/if}
			</select>
			<div class="buttons">
				<button type="submit" disabled={busy || !text.trim()}>Generate</button>
				<button type="button" class="secondary" disabled={!busy && !wav} onclick={() => { stop(); replay?.pause(); }}>Stop playback</button>
			</div>
		</div>
		<p class="hint">{mode === 'browser' ? 'The first browser run downloads the model. Later runs reuse the browser cache.' : 'Uses the Phonon service on port 8184. Models and all 37 voices are stored locally.'}</p>
	</form>

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
			<a class="download" href={wav} download={`phonon-31m-${recordedMode}-${recordedVoice.toLowerCase()}.wav`}>Download WAV ↓</a>
		{/if}
		<p class="hint">Time to first audio includes setup. Total generation measures synthesis after setup.</p>
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
	section { margin-top: 16px; }
	.row, .controls, .buttons, .examples { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
	.row { justify-content: space-between; margin-bottom: 8px; }
	.row span, .hint { color: #bcbfda; font-size: .82rem; }
	textarea, select, button { font: inherit; border-radius: 8px; padding: 8px 12px; }
	textarea, select { border: 1px solid #5a6084; background: #121524; color: inherit; }
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
	.error { color: #ffb4ad; overflow-wrap: anywhere; }
</style>

<script lang="ts">
	import { onDestroy } from 'svelte';
	import { receiveAudio } from '#lib/stream-audio.js';
	import { runOnnx } from '#lib/onnx/client.js';
	import models from '#lib/models.json';

	const voices = ['alba', 'anna', 'marius'];
	const onnxVoices = ['alba', 'marius', 'azelma', 'cosette', 'eponine', 'fantine', 'javert', 'jean'];
	const onnxModel = { id: 'english_18m_refl_2026-10', size: '18.2M', label: 'ONNX · WASM' };
	let worker: Worker | undefined;
	let text = $state('The rain had stopped by morning. Across the river, a small café was already open, and the smell of fresh bread drifted down the street.');
	let voice = $state('alba');
	let referenceFiles = $state<FileList>();
	let running = $state(false);
	let mode = $state<'standard' | 'streaming' | 'onnx'>('standard');
	let streaming = $derived(mode !== 'standard');
	let availableModels = $derived(mode === 'onnx' ? [onnxModel] : models);
	let availableVoices = $derived(mode === 'onnx' ? onnxVoices : voices);

	function selectMode(next: typeof mode) {
		if (next === mode) return;
		clearResults();
		mode = next;
		if (!availableVoices.includes(voice) && voice !== 'clone') voice = 'alba';
	}
	let context: AudioContext | undefined;
	let canRun = $derived(
		!running && !!text.trim() && (voice !== 'clone' || !!referenceFiles?.[0])
	);
	let selected = $state(models[0].id);
	let results = $state<Record<string, {
		audio?: string; error?: string; setup?: number; generation?: number;
		duration?: number; ttfa?: number; speed?: number; load?: string; voice: string;
	}>>({});
	let pending: AbortController | undefined;

	function clearResults() {
		if (context) void context.close().catch(() => {});
		context = undefined;
		for (const result of Object.values(results)) {
			if (result.audio) URL.revokeObjectURL(result.audio);
		}
		results = {};
	}

	onDestroy(() => {
		pending?.abort();
		worker?.terminate();
		clearResults();
	});

	async function run(ids: string[]) {
		if (!canRun) return;
		const reference = voice === 'clone' ? referenceFiles?.[0] : undefined;
		const input = { text: text.trim(), voice };
		const controller = new AbortController();
		pending = controller;
		running = true;
		clearResults();
		const playback = { next: 0 };
		try {
			if (streaming) {
				context = new AudioContext({ latencyHint: 'interactive' });
				await context.resume();
			}
			for (const model of ids) {
				try {
					if (mode === 'onnx' && context) {
						worker ??= new Worker(new URL('../lib/onnx/pocket.worker.ts', import.meta.url), { type: 'module' });
						const result = await runOnnx(worker, input, reference, context, controller.signal, (ttfa) => {
							results[model] = { voice: input.voice, ttfa };
						});
						if (controller.signal.aborted) break;
						results[model] = {
							...result,
							audio: URL.createObjectURL(result.audio),
							ttfa: results[model]?.ttfa,
							speed: result.duration * 1000 / result.generation,
							voice: input.voice
						};
						continue;
					}
					let body: FormData | string;
					if (reference) {
						body = new FormData();
						body.set('reference', reference);
						body.set('text', input.text);
						body.set('model', model);
					} else {
						body = JSON.stringify({ ...input, model });
					}
					const started = performance.now();
					const response = await fetch(streaming ? '/api/synthesize?stream=true' : '/api/synthesize', {
						method: 'POST',
						headers: reference ? undefined : { 'content-type': 'application/json' },
						body,
						signal: controller.signal
					});
					if (!response.ok) {
						const body = await response.json().catch(() => null);
						throw new Error(typeof body?.detail === 'string' ? body.detail : `Synthesis failed (HTTP ${response.status}).`);
					}
					const { audio, duration, generation } = context
						? await receiveAudio(response, context, started, playback, (ttfa) => {
							results[model] = { voice: input.voice, ttfa };
						})
						: {
							audio: await response.blob(),
							duration: Number(response.headers.get('x-audio-seconds')),
							generation: Number(response.headers.get('x-generation-ms'))
						};
					if (controller.signal.aborted) break;
					results[model] = {
						audio: URL.createObjectURL(audio),
						ttfa: results[model]?.ttfa,
						setup: Number(response.headers.get('x-setup-ms')),
						generation,
						duration,
						speed: generation > 0 ? duration * 1000 / generation : undefined,
						load: response.headers.get('x-model-load') ?? undefined,
						voice: input.voice
					};
				} catch (error) {
					if (controller.signal.aborted) break;
					if (mode === 'onnx') { worker?.terminate(); worker = undefined; }
					results[model] = { voice: input.voice, error: error instanceof Error ? error.message : 'Synthesis failed.' };
				}
			}
		} catch {
			results[ids[0]] = { voice: input.voice, error: 'Could not start browser audio.' };
		} finally {
			running = false;
			pending = undefined;
		}
	}
</script>

<svelte:head>
	<title>Pocket TTS Lab</title>
	<meta name="description" content="Compare the original English Pocket TTS model and its smaller variants." />
</svelte:head>

<main>
	<header>
		<p class="eyebrow"><span></span> LOCAL CPU · ENGLISH</p>
		<h1>Pocket TTS Lab</h1>
		<p>Compare Kyutai’s original 100M model with its three smaller variants.</p>
	</header>

	<nav aria-label="Synthesis mode">
		<button type="button" class:secondary={mode !== 'standard'} aria-pressed={mode === 'standard'} disabled={running}
			onclick={() => selectMode('standard')}>Standard</button>
		<button type="button" class:secondary={mode !== 'streaming'} aria-pressed={mode === 'streaming'} disabled={running}
			onclick={() => selectMode('streaming')}>Streaming</button>
		<button type="button" class:secondary={mode !== 'onnx'} aria-pressed={mode === 'onnx'} disabled={running}
			onclick={() => selectMode('onnx')}>ONNX Streaming</button>
	</nav>

	<form onsubmit={(event) => { event.preventDefault(); void run([mode === 'onnx' ? onnxModel.id : selected]); }}>
		{#if mode === 'onnx'}
			<p class="onnx-note">Runs in your browser with WebAssembly. The first run downloads model files.</p>
		{/if}
		<div class="label-row">
			<label for="text">Text to speak</label>
			<span>{text.length} / 1000</span>
		</div>
		<textarea id="text" bind:value={text} maxlength="1000" rows="4" required disabled={running}></textarea>
		<div class="controls">
			<label>Model
				{#if mode === 'onnx'}
					<select disabled><option>18.2M REFL</option></select>
				{:else}
				<select bind:value={selected} disabled={running}>
					{#each models as model (model.id)}<option value={model.id}>{model.size} parameters</option>{/each}
				</select>
				{/if}
			</label>
			<label>Voice
				<select bind:value={voice} onchange={() => { referenceFiles = undefined; }} disabled={running}>
					{#each availableVoices as name (name)}<option value={name}>{name}</option>{/each}
					<option value="clone">Clone from audio</option>
				</select>
			</label>
			<div class="buttons">
				<button type="submit" disabled={!canRun}>Generate</button>
				{#if mode !== 'onnx'}
					<button type="button" class="secondary" onclick={() => run(models.map((model) => model.id))} disabled={!canRun}>Compare all</button>
				{/if}
			</div>
		</div>
		{#if voice === 'clone'}
			<div class="reference">
				<label for="reference">Reference audio</label>
				<input id="reference" type="file" accept=".wav,.m4a,audio/wav,audio/mp4,audio/x-m4a" bind:files={referenceFiles} disabled={running} aria-describedby="reference-help" />
				<p id="reference-help">Choose a clear WAV or M4A speech sample up to 10 MB. The lab uses the first 30 seconds and processes the file on this Mac.</p>
			</div>
		{/if}
	</form>

	<section aria-labelledby="results">
		<div class="label-row">
			<h2 id="results">Model results</h2>
			<span role="status">{running ? (mode === 'onnx' ? 'Loading model or generating audio…' : 'Generating audio…') : 'Generation speed excludes setup'}</span>
		</div>
		<div class="cards">
			{#each availableModels as model (model.id)}
				{@const result = results[model.id]}
				<article>
					<p class="eyebrow">{model.label}</p>
					<h3>{model.size}</h3>
					<p class="model-id">{model.id}</p>
					{#if result?.audio}
						<audio controls src={result.audio} aria-label={`${model.size} audio`}></audio>
						<dl>
							{#if result.ttfa !== undefined}<div><dt>TTFA</dt><dd>{result.ttfa.toFixed(0)} ms</dd></div>{/if}
							<div><dt>{streaming ? 'Total generation' : 'Generation'}</dt><dd>{result.generation?.toFixed(0)} ms</dd></div>
							<div><dt>Audio</dt><dd>{result.duration?.toFixed(2)} s</dd></div>
							<div><dt>Speed</dt><dd>{result.speed?.toFixed(1) ?? '—'}× real time</dd></div>
							<div><dt>Setup ({result.load})</dt><dd>{result.setup?.toFixed(0)} ms</dd></div>
						</dl>
						<a class="download" href={result.audio} download={`pocket-${model.size}-${result.voice}.wav`}>Download WAV ↓</a>
					{:else if result?.ttfa !== undefined}
						<p role="status">TTFA {result.ttfa.toFixed(0)} ms · streaming</p>
					{:else if result?.error}
						<p class="error" role="alert">{result.error}</p>
					{:else}
						<p class="empty">Awaiting audio</p>
					{/if}
				</article>
			{/each}
		</div>
	</section>
</main>

<style>
	:global(*) { box-sizing: border-box; }
	:global(body) { margin: 0; background: #101715; color: #e9f2eb; font: 16px/1.5 system-ui, sans-serif; color-scheme: dark; }
	main { width: min(1080px, 100%); margin: auto; padding: clamp(24px, 6vw, 72px) clamp(16px, 4vw, 40px); }
	header { margin-bottom: 32px; }
	h1 { font-size: clamp(2.5rem, 7vw, 5.5rem); letter-spacing: -.065em; line-height: 1; margin: 12px 0; }
	header > p:last-child { color: #b4c6bb; margin: 0; }
	.eyebrow, .model-id { font: .8rem/1.4 ui-monospace, monospace; color: #a6c1b0; }
	.eyebrow { text-transform: uppercase; letter-spacing: .12em; margin: 0; }
	.eyebrow span { display: inline-block; background: #a8efaa; border-radius: 50%; width: 8px; height: 8px; margin-right: 5px; }
	nav { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
	form, article { border: 1px solid #34453d; background: #18231e; border-radius: 16px; padding: 20px; }
	.label-row, .controls, .buttons { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
	.label-row { justify-content: space-between; margin-bottom: 8px; }
	.label-row span { font-size: .8rem; color: #b4c6bb; }
	textarea, select, button { font: inherit; border-radius: 8px; padding: 8px 12px; }
	textarea, select { background: #101715; color: inherit; border: 1px solid #42594e; }
	textarea { width: 100%; display: block; resize: vertical; min-height: 88px; }
	.controls { margin-top: 16px; }
	.controls label { display: flex; align-items: center; gap: 8px; }
	select { text-transform: capitalize; }
	.buttons { margin-left: auto; }
	button { background: #a8efaa; color: #122719; border: 1px solid #a8efaa; font-weight: 650; cursor: pointer; }
	button.secondary { background: transparent; color: #bff2c1; border-color: #62886c; }
	button:disabled { opacity: .5; cursor: default; }
	.onnx-note { margin: 0 0 12px; color: #b4c6bb; }
	.reference { display: grid; gap: 6px; margin-top: 16px; }
	.reference p { margin: 0; color: #b4c6bb; font-size: .85rem; }
	section { margin-top: 30px; }
	h2 { font-size: 1rem; margin: 0; }
	.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr)); gap: 12px; }
	article { min-width: 0; }
	h3 { font-size: 2rem; margin: 1px 0; }
	.model-id { margin: 0 0 14px; overflow-wrap: anywhere; }
	.empty { color: #b4c6bb; margin-bottom: 0; }
	.error { color: #ffad9b; margin-bottom: 0; }
	audio { width: 100%; height: 42px; }
	dl { margin: 8px 0 12px; }
	dl div { display: flex; justify-content: space-between; gap: 6px; font-size: .85rem; }
	dt { color: #b4c6bb; }
	dd { margin: 0; font-variant-numeric: tabular-nums; }
	.download { color: #bff2c1; font-size: .85rem; }
</style>

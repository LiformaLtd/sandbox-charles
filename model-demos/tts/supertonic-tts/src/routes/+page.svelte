<script lang="ts">
 import { onDestroy } from 'svelte';
 import BrowserWorker from '../lib/browser.worker?worker';
 import { voices, languages, type Packet, type Settings } from '../lib/model';
 import { makeWav, playChunk } from '../lib/audio';
 const prompts = {
  'Pocket TTS comparison': 'The rain had stopped by morning. Across the river, a small café was already open, and the smell of fresh bread drifted down the street.',
  'Numbers and units': 'Order number 3047 arrives on Thursday. The parcel weighs 2.75 kilograms, and delivery costs £12.50.',
  'Clear consonants': 'She sells seashells by the seashore. Six slick swans swam swiftly past the stone steps.'
 };
 const languageNames = new Intl.DisplayNames(['en'], { type: 'language' });
 let mode = $state<'server' | 'browser'>('server');
 let text = $state(prompts['Pocket TTS comparison']);
 let voice = $state('F1');
 let language = $state('en');
 let steps = $state(5);
 let speed = $state(1.05);
 let busy = $state(false);
 let status = $state('Ready');
 let failure = $state('');
 let first = $state<number>();
 let setup = $state<number>();
 let generation = $state<number>();
 let duration = $state<number>();
 let wav = $state('');
 let recordedVoice = $state('');
 let recordedMode = $state('');
 let replay = $state<HTMLAudioElement>();
 let context: AudioContext | undefined;
 let controller: AbortController | undefined;
 let worker: Worker | undefined;
 let rejectWorker: ((e: Error) => void) | undefined;
 function stop() {
  controller?.abort();
  if (busy && worker) { worker.terminate(); worker = undefined; rejectWorker?.(new DOMException('Stopped', 'AbortError')); }
  rejectWorker = undefined;
  if (context) { void context.close(); context = undefined; }
 }
 onDestroy(() => { stop(); worker?.terminate(); if (wav) URL.revokeObjectURL(wav); });
 async function generate() {
  if (busy || !text.trim()) return;
  stop(); replay?.pause();
  if (wav) URL.revokeObjectURL(wav);
  wav = ''; failure = ''; first = setup = generation = duration = undefined;
  busy = true; status = 'Preparing…'; recordedMode = mode === 'server' ? 'Local server' : 'Browser ONNX';
  const run = new AbortController(); controller = run;
  const started = performance.now();
  const chunks: ArrayBuffer[] = [];
  let sampleRate = 0;
  let samples = 0;
  let completed = false;
  try {
   const audio = new AudioContext({ latencyHint: 'interactive' }); context = audio; await audio.resume();
   const playback = { next: audio.currentTime };
   const accept = (packet: Packet) => {
    run.signal.throwIfAborted();
    if (packet.error) throw new Error(packet.error);
    if (packet.status) status = packet.status;
    if (packet.setup !== undefined) setup = packet.setup;
    if (packet.generation !== undefined) generation = packet.generation;
    if (packet.done) completed = true;
    if (packet.pcm) {
     if (!packet.sampleRate) throw new Error('Missing audio sample rate.');
     sampleRate = packet.sampleRate;
     playChunk(audio, playback, packet.pcm, sampleRate);
     if (first === undefined) first = performance.now() - started;
     samples += packet.pcm.length; chunks.push(packet.pcm.buffer);
    }
   };
   const settings: Settings = { text: text.trim(), voice, language, steps, speed };
   if (mode === 'browser') {
    worker ??= new BrowserWorker();
    await new Promise<void>((resolve, reject) => {
     rejectWorker = reject;
     worker!.onerror = event => { worker?.terminate(); worker = undefined; reject(new Error(event.message || 'Browser worker failed.')); };
     worker!.onmessage = ({ data }: MessageEvent<Packet>) => {
      try { accept(data); if (data.done) resolve(); } catch (e) { reject(e); }
     };
     worker!.postMessage(settings);
    });
   } else {
    const response = await fetch('/api/synthesize', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings), signal: run.signal });
    if (!response.ok || !response.body) throw new Error(`Local synthesis request failed (${response.status}).`);
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let pending = '';
    try {
     while (true) {
      const { value, done } = await reader.read(); if (done) break;
      const lines = (pending + value).split('\n'); pending = lines.pop() || '';
      for (const line of lines) {
       if (!line) continue;
       const packet = JSON.parse(line);
       if (packet.pcm) packet.pcm = new Float32Array(Uint8Array.from(atob(packet.pcm), c => c.charCodeAt(0)).buffer);
       accept(packet);
      }
     }
     if (pending) throw new Error('Incomplete audio stream.');
    } finally { await reader.cancel(); }
   }
   run.signal.throwIfAborted();
   if (!completed || !samples || generation === undefined) throw new Error('Audio generation ended before completion.');
   duration = samples / sampleRate;
   wav = URL.createObjectURL(makeWav(chunks, samples, sampleRate));
   recordedVoice = voice; status = 'Generation complete';
  } catch (e) {
   if (run.signal.aborted) status = 'Stopped';
   else { console.error('Supertonic generation failed', e); failure = e instanceof Error ? e.message : String(e); status = 'Generation failed'; }
   stop();
  } finally { busy = false; controller = undefined; rejectWorker = undefined; }
 }
</script>

<svelte:head>
	<title>Supertonic 3 Lab</title>
	<meta name="description" content="Compare Supertonic 3 with native server and browser ONNX inference." />
</svelte:head>

<main>
	<header>
		<p class="eyebrow">99M · 31 LANGUAGES · ONNX</p>
		<h1>Supertonic 3 Lab</h1>
		<p>Compare clarity and speed on your Mac and in your browser.</p>
	</header>

	<nav aria-label="Inference mode">
  <button class:secondary={mode !== 'server'} disabled={busy} onclick={() => mode = 'server'} aria-pressed={mode === 'server'}>Local server</button>
  <button class:secondary={mode !== 'browser'} disabled={busy} onclick={() => mode = 'browser'} aria-pressed={mode === 'browser'}>Browser ONNX</button>
 </nav>
 <p class="hint mode-note">{mode === 'server' ? 'Native ONNX Runtime on this Mac. Text stays on the local server.' : 'ONNX Runtime Web with WASM in a browser worker. Synthesis runs on your device.'}</p>
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
			<label for="voice">Voice</label>
			<select id="voice" bind:value={voice} disabled={busy}>
    {#each voices as name (name)}<option value={name}>{name.startsWith('F') ? 'Female' : 'Male'} · {name}</option>{/each}
			</select>
			<label for="language">Language</label>
   <select id="language" bind:value={language} disabled={busy}>{#each languages as code (code)}<option value={code}>{languageNames.of(code)}</option>{/each}</select>
   <label for="steps">Denoising steps</label>
   <input id="steps" type="number" min="1" max="20" step="1" bind:value={steps} disabled={busy} />
   <label for="speed">Speed</label>
   <input id="speed" type="number" min="0.5" max="2" step="0.05" bind:value={speed} disabled={busy} />
   <div class="buttons">
				<button type="submit" disabled={busy || !text.trim()}>Generate</button>
				<button type="button" class="secondary" disabled={!busy && !wav} onclick={() => { stop(); replay?.pause(); }}>Stop playback</button>
			</div>
		</div>
		<p class="hint">Both modes use the same standard model and voices. More denoising steps trade speed for quality. The first run downloads about 400 MB of model files.</p>
	</form>

	<section aria-labelledby="result">
		<div class="row"><h2 id="result">{recordedMode || (mode === 'server' ? 'Local server' : 'Browser ONNX')} result</h2><span role="status">{status}</span></div>
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
			<a class="download" href={wav} download={`supertonic-3-${recordedVoice.toLowerCase()}.wav`}>Download WAV ↓</a>
		{/if}
		<p class="hint">Playback starts after the first text segment is generated. TTFA includes setup; total generation measures synthesis of all segments, including gaps, after setup.</p>
	</section>
</main>

<style>
	:global(*) { box-sizing: border-box; }
	:global(body) { margin: 0; background: #121524; color: #f1f1ff; color-scheme: dark; font: 16px/1.5 system-ui, sans-serif; }
	main { max-width: 880px; margin: auto; padding: clamp(24px, 7vw, 78px) clamp(16px, 4vw, 40px); }
	header { margin-bottom: 28px; }
 nav { display: flex; gap: 8px; }
 .mode-note { margin: 10px 0 20px; }
 input { width: 85px; }
	h1 { margin: 6px 0; font-size: clamp(2.5rem, 8vw, 5rem); letter-spacing: -.06em; line-height: 1.1; }
	header p { margin: 0; color: #bcbfda; }
	header .eyebrow { color: #bfb0ff; font: .8rem ui-monospace, monospace; letter-spacing: .13em; }
	form, section { border: 1px solid #3c4061; background: #1b2035; border-radius: 16px; padding: clamp(14px, 4vw, 24px); }
	section { margin-top: 16px; }
	.row, .controls, .buttons, .examples { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
	.row { justify-content: space-between; margin-bottom: 8px; }
	.row span, .hint { color: #bcbfda; font-size: .82rem; }
	textarea, select, input, button { font: inherit; border-radius: 8px; padding: 8px 12px; }
	textarea, select, input { border: 1px solid #5a6084; background: #121524; color: inherit; }
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

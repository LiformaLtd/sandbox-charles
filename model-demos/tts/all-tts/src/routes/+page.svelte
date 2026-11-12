<script lang="ts">
 import { onDestroy } from 'svelte';
 import { families, modeLabels, familyById, modelById, voiceOptions, type Family, type Mode } from '../lib/catalogue';
 import { languages } from '../lib/supertonic/model';
 import { Runtimes } from '../lib/runtimes';
 const runtime = new Runtimes();
 const prompts = {
  'Comparison passage': 'The rain had stopped by morning. Across the river, a small café was already open, and the smell of fresh bread drifted down the street.',
  'Numbers and units': 'Order number 3047 arrives on Thursday. The parcel weighs 2.75 kilograms, and delivery costs £12.50.',
  'Clear consonants': 'She sells seashells by the seashore. Six slick swans swam swiftly past the stone steps.'
 };
 const languageNames = new Intl.DisplayNames(['en'], { type: 'language' });
 let family = $state<Family>('phonon');
 let model = $state('phonon-31m-en');
 let mode = $state<Mode>('browser');
 let voice = $state('Easton');
 let language = $state('en');
 let text = $state(prompts['Comparison passage']);
 let steps = $state(5);
 let speed = $state(1.05);
 let reference = $state<File>();
 let busy = $state(false);
 let status = $state('Ready');
 let failure = $state('');
 let first = $state<number>();
 let setup = $state<number>();
 let generation = $state<number>();
 let duration = $state<number>();
 let wav = $state('');
 let recordedName = $state('');
 let recordedFile = $state('speech');
 let replay = $state<HTMLAudioElement>();
 let controller: AbortController | undefined;
 let context: AudioContext | undefined;
 const selectedFamily = $derived(familyById(family));
 const selectedModel = $derived(modelById(family, model));
 const voiceGroups = $derived(voiceOptions(family, mode));
 const isClone = $derived(family === 'pocket' && voice === '__clone');
 function ensureVoice() {
  if (!voiceOptions(family, mode).flatMap(g => g.voices).some(v => v.value === voice)) voice = family === 'phonon' ? 'Easton' : voiceOptions(family, mode)[0].voices[0].value;
 }
 function chooseFamily(value: Family) {
  family = value; model = familyById(value).models[0].id; mode = modelById(value, model).modes[0]; language = 'en'; ensureVoice();
 }
 function chooseModel(value: string) {
  model = value; if (!selectedModel.modes.includes(mode)) mode = selectedModel.modes[0]; ensureVoice();
 }
 function stop() {
  controller?.abort(); if (busy) runtime.cancel();
  if (context) { void context.close(); context = undefined; }
  replay?.pause();
 }
 onDestroy(() => { stop(); runtime.dispose(); if (wav) URL.revokeObjectURL(wav); });
 async function generate() {
  if (busy || !text.trim()) return;
  if (isClone && (!reference || reference.size > 10 * 1024 * 1024 || !reference.size)) { failure = 'Choose a WAV or M4A file up to 10 MB.'; return; }
  stop(); if (wav) URL.revokeObjectURL(wav); wav = ''; failure = ''; first = setup = generation = duration = undefined;
  const run = new AbortController(); controller = run; busy = true; status = 'Preparing…';
  const started = performance.now();
  recordedName = `${selectedFamily.label} · ${selectedModel.label} · ${modeLabels[mode]} · ${isClone ? 'Cloned voice' : voice}`;
  recordedFile = `${family}-${model}-${isClone ? 'clone' : voice}`.toLowerCase();
  try {
   const audio = new AudioContext({ latencyHint: 'interactive' }); context = audio; await audio.resume();
   const result = await runtime.run({ family, model, mode, text: text.trim(), voice, language, steps, speed, reference: isClone ? reference : undefined }, audio, run.signal,
    value => status = value, value => setup = value, () => { if (first === undefined) first = performance.now() - started; });
   run.signal.throwIfAborted();
   setup = result.setup; generation = result.generation; duration = result.duration;
   wav = URL.createObjectURL(result.audio); status = 'Generation complete';
  } catch (e) {
   if (run.signal.aborted) status = 'Stopped';
   else { console.error('TTS comparison generation failed', { family, model, mode }, e); failure = e instanceof Error ? e.message : String(e); status = 'Generation failed'; }
   stop();
  } finally { busy = false; controller = undefined; }
 }
</script>

<svelte:head>
 <title>All TTS Lab</title>
 <meta name="description" content="Compare Phonon, Supertonic and Pocket TTS speed and clarity." />
</svelte:head>
<main>
 <header>
  <p class="eyebrow">THREE FAMILIES · ONE TEST BENCH</p>
  <h1>All TTS Lab</h1>
  <p>Choose a family and model. Compare how quickly and clearly it speaks.</p>
 </header>
 <form onsubmit={event => { event.preventDefault(); void generate(); }}>
  <div class="selectors">
   <label>Model family
    <select value={family} disabled={busy} onchange={e => chooseFamily(e.currentTarget.value as Family)}>{#each families as f (f.id)}<option value={f.id}>{f.label}</option>{/each}</select>
   </label>
   <label>Model
    <select value={model} disabled={busy} onchange={e => chooseModel(e.currentTarget.value)}>{#each selectedFamily.models as m (m.id)}<option value={m.id}>{m.label}</option>{/each}</select>
   </label>
   <label>Run on
    <select value={mode} disabled={busy || selectedModel.modes.length === 1} onchange={e => { mode = e.currentTarget.value as Mode; ensureVoice(); }}>{#each selectedModel.modes as m (m)}<option value={m}>{modeLabels[m]}</option>{/each}</select>
   </label>
  </div>
  <p class="hint model-note">{selectedModel.licence}. {family === 'phonon' ? (mode === 'browser' ? 'Browser WASM · English · Q8.' : mode === 'native' ? 'Local native Rust runtime · CPU · Q8.' : 'Local MLX runtime · Apple GPU · standard weights.') : family === 'supertonic' ? 'Standard ONNX · 31 languages.' : mode === 'browser' ? 'Browser ONNX · 18.2M REFL · INT8 main graph.' : 'Local Python inference · English.'}</p>
  <div class="row"><label for="speech">Text to speak</label><span>{text.length} / 1000</span></div>
  <textarea id="speech" bind:value={text} rows="4" maxlength="1000" required disabled={busy}></textarea>
  <div class="examples" aria-label="Example prompts">{#each Object.entries(prompts) as [name, example] (name)}<button type="button" class="secondary" disabled={busy} onclick={() => text = example}>{name}</button>{/each}</div>
  <div class="controls">
   <label for="voice">Voice</label>
   <select id="voice" bind:value={voice} disabled={busy}>{#each voiceGroups as group (group.group)}<optgroup label={group.group}>{#each group.voices as v (v.value)}<option value={v.value}>{v.label}</option>{/each}</optgroup>{/each}</select>
   {#if family === 'supertonic'}
    <label for="language">Language</label><select id="language" bind:value={language} disabled={busy}>{#each languages as code (code)}<option value={code}>{languageNames.of(code)}</option>{/each}</select>
    <label for="steps">Denoising steps</label><input id="steps" type="number" min="1" max="20" step="1" bind:value={steps} disabled={busy} />
    <label for="speed">Speed</label><input id="speed" type="number" min="0.5" max="2" step="0.05" bind:value={speed} disabled={busy} />
   {/if}
   {#if isClone}
    <label for="reference">Reference audio</label><input id="reference" type="file" accept=".wav,.m4a,audio/wav,audio/mp4,audio/x-m4a" disabled={busy} onchange={e => reference = e.currentTarget.files?.[0]} />
   {/if}
  </div>
  <div class="buttons actions">
   <button type="submit" disabled={busy || !text.trim() || (isClone && !reference)}>Generate</button>
   <button type="button" class="secondary" disabled={!busy && !wav} onclick={stop}>Stop playback</button>
  </div>
  <p class="hint">{family === 'phonon' && mode !== 'browser' ? 'Uses the local Phonon service on port 8184. Its model and all 37 voices are already stored locally.' : family === 'supertonic' ? 'Playback starts after each text segment is ready. The first model download is about 400 MB.' : 'Audio streams as it is generated. First runs include model downloads; later runs reuse cached files.'}</p>
 </form>
 <section aria-labelledby="result">
  <div class="row"><h2 id="result">Streaming result</h2><span role="status">{status}</span></div>
  {#if recordedName}<p class="hint result-name">{recordedName}</p>{/if}
  {#if failure}<p class="error" role="alert">{failure}</p>{/if}
  <dl>
   <div><dt>Time to first audio</dt><dd>{first === undefined ? '—' : `${Math.round(first)} ms`}</dd></div>
   <div><dt>Total generation</dt><dd>{generation === undefined ? '—' : `${Math.round(generation)} ms`}</dd></div>
   <div><dt>Setup</dt><dd>{setup === undefined ? '—' : `${Math.round(setup)} ms`}</dd></div>
   <div><dt>Audio duration</dt><dd>{duration === undefined ? '—' : `${duration.toFixed(2)} s`}</dd></div>
   <div><dt>Generation speed</dt><dd>{duration && generation ? `${(duration * 1000 / generation).toFixed(1)}× real time` : '—'}</dd></div>
  </dl>
  {#if wav}<audio bind:this={replay} controls src={wav} aria-label="Replay generated speech" onplay={() => { if (context) { void context.close(); context = undefined; } }}></audio><a class="download" href={wav} download={`${recordedFile}.wav`}>Download WAV ↓</a>{/if}
  <p class="hint">TTFA includes setup and first audio. Total generation measures synthesis after setup. Voices and hardware runtimes differ between families.</p>
 </section>
</main>
<style>
	:global(*) { box-sizing: border-box; }
	:global(body) { margin: 0; background: #121524; color: #f1f1ff; color-scheme: dark; font: 16px/1.5 system-ui, sans-serif; }
	main { max-width: 880px; margin: auto; padding: clamp(24px, 7vw, 78px) clamp(16px, 4vw, 40px); }
	header { margin-bottom: 28px; }


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

 .selectors { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
 .selectors label { display: flex; flex-direction: column; gap: 6px; }
 .selectors select { width: 100%; }
 .model-note { margin: 12px 0 22px; }
 .actions { margin: 18px 0 0; }
 .result-name { margin: 0 0 14px; }
 input[type='file'] { width: min(100%, 350px); }
</style>

<script lang="ts">
  import { onMount } from 'svelte';
  import { concatPcm, encodeWav } from 'phonon-tts';
  import { createJames } from '../lib/studio/rig.js';
  import { openPhononStream, releasePhonon } from '../lib/studio/phonon.js';
  import { blinkAt, idleHeadAt } from '../lib/studio/idle-motion.js';
  import idleTrack from '../lib/studio/idle-head-track.json';
  import DictionaryWorker from '../lib/dictionary/dictionary.worker.js?worker';
  import type { prepareTranscript } from '../lib/alignment/engine.ts';
  import { CHANNELS, FPS, SAMPLE_RATE } from '../lib/dictionary/index.mjs';
  import { dialogueSamples } from '../lib/dialogue-samples';
  import '../lib/studio/style.css';

  interface AnimationResult {
    frames: Float32Array;
    phonemeCount: number;
    coverage: Record<'triple' | 'pair' | 'single' | 'fallback' | 'miss' | 'approximate', number>;
    usage: Record<'triple' | 'pair' | 'single' | 'approximate', number>;
    joins: number;
    lookupMs: number;
    alignmentMs: number;
    samplingMs: number;
    corpus: 'pilot' | 'full' | 'targeted' | 'subtitles';
    pauseUsage?: { used: number; detected: number };
  }

  let sampleIndex = $state(0);
  let dialogue = $state<string>(dialogueSamples[0].text);
  const sampleTitle = $derived(dialogue === dialogueSamples[sampleIndex].text ? dialogueSamples[sampleIndex].title : 'Custom dialogue');
  let status = $state('Loading the James rig…');
  let error = $state('');
  let inputError = $state('');
  let strength = $state(60);
  let closeup = $state(false);
  let recordedPauses = $state(false);
  let ready = $state(false);
  let busy = $state(false);
  let details = $state<(Omit<AnimationResult, 'frames'> & { ttsMs: number }) | null>(null);
  let duration = $state(0);
  let viewport: HTMLDivElement;
  let audio: HTMLAudioElement;
  let rig: Awaited<ReturnType<typeof createJames>> | undefined;
  let worker: Worker | undefined;
  let requestId = 0;
  let frameViews: Float32Array[] = [];
  let audioUrl: string | undefined;
  let run: AbortController | undefined;
  let pending: ((reason: Error) => void) | undefined;

  const mobile = () => /iPhone|iPad|iPod/.test(navigator.userAgent)
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

  function askWorker<T>(type: 'prepare' | 'animate', text: string | ReturnType<typeof prepareTranscript>, pcm?: Float32Array): Promise<T> {
    const activeWorker = worker ??= new DictionaryWorker();
    return new Promise((resolve, reject) => {
      const id = ++requestId;
      pending = reject;
      activeWorker.onmessage = ({ data }) => {
        if (data.id !== id) return;
        if (data.status) {
          status = data.status;
          return;
        }
        pending = undefined;
        if (data.error) reject(new Error(data.error));
        else resolve(data.value);
      };
      activeWorker.onerror = (event) => {
        pending = undefined;
        activeWorker.terminate();
        worker = undefined;
        reject(new Error(event.message || 'Could not start the alignment worker.'));
      };
      activeWorker.postMessage({ id, type, text, pcm, recordedPauses }, pcm ? [pcm.buffer] : []);
    });
  }

  function setView(next: boolean) {
    closeup = next;
    rig?.view(next);
  }

  onMount(() => {
    let disposed = false;
    let animation = 0;
    let started: number | undefined;
    let previousTick = -1;
    async function start() {
      try {
        const viewer = await createJames(viewport, { compact: true });
        rig = viewer;
        if (disposed) {
          viewer.dispose();
          return;
        }
        ready = true;
        status = 'Ready to generate speech.';
        function draw(now: number) {
          started ??= now;
          const elapsed = (now - started) / 1000;
          const tick = Math.floor(elapsed * FPS);
          if (tick !== previousTick) {
            const index = !audio.ended && frameViews.length
              ? Math.min(frameViews.length - 1, Math.floor(audio.currentTime * FPS)) : -1;
            const rotation = strength ? idleHeadAt(idleTrack, elapsed).map((angle: number) => angle * strength / 100) : undefined;
            viewer.pose(frameViews[index], blinkAt(elapsed), rotation);
            viewport.dataset.frame = index < 0 ? 'neutral' : String(index);
            previousTick = tick;
          }
          viewer.draw();
          animation = requestAnimationFrame(draw);
        }
        animation = requestAnimationFrame(draw);
      } catch (cause) {
        if (!disposed) error = messageOf(cause);
      }
    }
    start();
    return () => {
      disposed = true;
      cancelAnimationFrame(animation);
      run?.abort();
      pending?.(new DOMException('Cancelled', 'AbortError'));
      worker?.terminate();
      rig?.dispose();
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      releasePhonon().catch(console.error);
    };
  });

  function stop() {
    run?.abort();
    if (pending) {
      pending(new DOMException('Cancelled', 'AbortError'));
      pending = undefined;
      worker?.terminate();
      worker = undefined;
    }
  }

  function stepDialogue(direction: number) {
    if (busy) return;
    sampleIndex = (sampleIndex + direction + dialogueSamples.length) % dialogueSamples.length;
    dialogue = dialogueSamples[sampleIndex].text;
    inputError = '';
    error = '';
    audio?.pause();
    audio?.removeAttribute('src');
    audio?.load();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = undefined;
    frameViews = [];
    details = null;
    duration = 0;
    if (ready) status = 'Ready to generate speech.';
  }

  const elapsed = (milliseconds: number) => milliseconds < 1000
    ? milliseconds.toFixed(1) + ' ms' : (milliseconds / 1000).toFixed(2) + ' s';

  const messageOf = (cause: unknown) => cause instanceof Error ? cause.message : String(cause);

  async function generate(event: SubmitEvent) {
    event.preventDefault();
    if (busy || !ready) return;
    const text = dialogue.trim();
    const current = new AbortController();
    const { signal } = current;
    run = current;
    busy = true;
    inputError = '';
    error = '';
    let phase = 'input';
    try {
      status = 'Checking the transcript…';
      const preparationStarted = performance.now();
      const transcript = await askWorker<ReturnType<typeof prepareTranscript>>('prepare', text);
      const preparationMs = performance.now() - preparationStarted;
      signal.throwIfAborted();
      phase = 'speech';
      audio.pause();
      status = 'Preparing Phonon…';
      const speechStarted = performance.now();
      const speech = await openPhononStream(text, (message: string) => { status = message; }, signal, mobile());
      const chunks = [];
      for await (const chunk of speech.chunks) chunks.push(chunk);
      await speech.done;
      signal.throwIfAborted();
      const pcm = concatPcm(chunks);
      if (pcm.length < 2) throw new Error('Phonon returned no usable audio.');
      const seconds = pcm.length / SAMPLE_RATE;
      const wav = new Blob([encodeWav(pcm, SAMPLE_RATE)], { type: 'audio/wav' });
      const ttsMs = performance.now() - speechStarted;
      status = 'Aligning phonemes with TinyAligner…';
      const { frames, ...stats } = await askWorker<AnimationResult>('animate', transcript, pcm);
      signal.throwIfAborted();
      frameViews = Array.from({ length: frames.length / CHANNELS }, (_, i) =>
        frames.subarray(i * CHANNELS, (i + 1) * CHANNELS));
      details = { ...stats, ttsMs, alignmentMs: preparationMs + stats.alignmentMs };
      duration = seconds;
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      audioUrl = URL.createObjectURL(wav);
      audio.src = audioUrl;
      status = 'Animation ready.';
      try { await audio.play(); }
      catch (cause) {
        if (!(cause instanceof DOMException && cause.name === 'NotAllowedError')) throw cause;
        status = 'Animation ready. Press play to listen.';
      }
    } catch (cause) {
      if (signal.aborted || (cause instanceof DOMException && cause.name === 'AbortError')) status = 'Generation cancelled.';
      else if (phase === 'input') {
        inputError = messageOf(cause);
        status = 'Check the dialogue and try again.';
      } else {
        error = messageOf(cause);
        status = 'Generation failed.';
      }
    } finally {
      busy = false;
      run = undefined;
    }
  }
</script>

<svelte:head><title>James · Facial Motion Studio</title></svelte:head>
<header class="masthead">
  <div class="brand"><span class="mark" aria-hidden="true">◉</span> Facial Motion Studio</div>
  <div class="badge"><span></span> Local preview</div>
</header>
<main class="studio">
  <div class="viewport" bind:this={viewport} data-frame="neutral">
    <div class="view-title"><span>RIG 01</span><strong>James</strong></div>
    <div class="view-tools" aria-label="Camera presets">
      <button type="button" aria-pressed={!closeup} onclick={() => setView(false)}>Face</button>
      <button type="button" aria-pressed={closeup} onclick={() => setView(true)}>Mouth close-up</button>
    </div>
    <div class="view-hint">Drag to orbit · Scroll to zoom</div>
  </div>
  <aside class="control-panel">
    <div class="eyebrow">TEXT TO ANIMATION</div>
    <h1>Make James speak.</h1>
    <p class="intro">Type a line and watch James move in sync with the generated speech.</p>
    <form onsubmit={generate}>
      <div class="voice-switch">Speech engine · Phonon</div>
      <div class="voice-switch">Animation model · Dictionary</div>
      <div class="dialogue-heading">
        <label for="speech">Dialogue</label>
        <div class="dialogue-navigation" role="group" aria-label="Sample dialogues">
          <button type="button" aria-label="Previous dialogue" disabled={busy} onclick={() => stepDialogue(-1)}>← Prev</button>
          <span>{sampleIndex + 1} / {dialogueSamples.length}</span>
          <button type="button" aria-label="Next dialogue" disabled={busy} onclick={() => stepDialogue(1)}>Next →</button>
        </div>
      </div>
      <p class="dialogue-sample" id="dialogue-sample" aria-live="polite">{sampleTitle}</p>
      <textarea id="speech" rows="5" maxlength="500" required bind:value={dialogue}
        oninput={() => { inputError = ''; }} aria-invalid={!!inputError}
        aria-describedby={inputError ? 'dialogue-sample dialogue-error' : 'dialogue-sample'}></textarea>
      {#if inputError}<p id="dialogue-error" class="status error" role="alert">{inputError}</p>{/if}
      <div class="input-meta"><span>Easton · generated in this browser</span><span>{dialogue.length} / 500</span></div>
      <label class="pause-pilot"><input type="checkbox" bind:checked={recordedPauses} disabled={busy} /> Recorded pauses (pilot)</label>
      <div class="head-motion-control">
        <label for="head-motion">Head motion</label>
        <span aria-hidden="true">{strength}%</span>
        <input type="range" id="head-motion" min="0" max="100" step="1" bind:value={strength} aria-valuetext={strength + '%'} />
      </div>
      <button type="submit" class="generate" disabled={!ready || busy}>Generate &amp; play <span aria-hidden="true">↗</span></button>
    </form>
    <p class="status" role="status" aria-live="polite">{status}</p>
    {#if error}<p class="status error" role="alert">{error}</p>{/if}
    {#if busy}
      <progress aria-label="Generation in progress"></progress>
      <button type="button" class="stop-stream" onclick={stop}>Cancel generation</button>
    {/if}
    <div class="playback" hidden={!details}>
      <div class="playback-label"><strong>Playback</strong><span>{duration.toFixed(2)} s</span></div>
      <audio bind:this={audio} controls preload="auto" aria-label="Generated speech"></audio>
    </div>
    {#if details}
      <div class="coverage" role="status" aria-live="polite">
        <p>{details.phonemeCount} phonemes in the speech</p>
        <p>{details.usage.triple} exact triples · {details.usage.approximate} approximate triples (P ↔ B) · {details.usage.pair} doubles · {details.usage.single} single phonemes</p>
        <p>Dictionary matches selected before blending.</p>
        <ol aria-label="Processing times">
          <li>Text to speech · {elapsed(details.ttsMs)}</li>
          <li>TinyAligner · {elapsed(details.alignmentMs)}</li>
          <li>Sampling · {elapsed(details.samplingMs)}</li>
        </ol>
        <p>Includes model and asset loading on first use.</p>
      </div>
      <details class="coverage">
        <summary>Match coverage</summary>
        <p>{details.coverage.triple + details.coverage.pair} exact ({details.coverage.triple} triples, {details.coverage.pair} doubles) · {details.coverage.approximate} approximate intervals · {details.coverage.fallback} single-phoneme fallback · {details.coverage.miss} misses</p>
        {#if details.coverage.single}<p>{details.coverage.single} isolated phoneme matches</p>{/if}
        <p>{details.joins} extended joins · {details.lookupMs.toFixed(1)} ms lookup and blend · {details.corpus} corpus</p>
        {#if details.pauseUsage}<p>Recorded pauses: {details.pauseUsage.used} of {details.pauseUsage.detected} used.</p>{/if}
      </details>
    {/if}
    <div class="notes"><span>James v3 mesh</span><span>60 fps · Dictionary</span></div>
  </aside>
</main>

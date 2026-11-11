<script lang="ts">
  import { onMount } from 'svelte';
  import { align, prepareText } from '../lib/alignment/engine';
  import { disposeSpeech, synthesize } from '../lib/speech';
  import type { AlignedWord } from '../lib/alignment/types';

  let text = $state('She sells seashells by the seashore.');
  let busy = $state(false);
  let status = $state('');
  let error = $state('');
  let audioUrl = $state('');
  let audio = $state<HTMLAudioElement>();
  let time = $state(0);
  let duration = $state(0);
  let waveform = $state('');
  let words = $state<AlignedWord[]>([]);
  let alignMs = $state(0);
  let phones = $derived(words.flatMap((word) => word.phones));
  let run: AbortController | undefined;

  onMount(() => {
    void generate();
    return () => {
      run?.abort();
      disposeSpeech();
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  });

  async function generate() {
    if (busy || !text.trim()) return;
    const prompt = text.trim();
    const { signal } = run = new AbortController();
    busy = true;
    error = '';
    words = [];
    time = 0;
    audio?.pause();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = '';
    try {
      status = 'Checking pronunciation…';
      const transcript = await prepareText(prompt);
      signal.throwIfAborted();
      const { samples, sampleRate, wav } = await synthesize(prompt, signal, (message) => status = message);
      status = 'Aligning phonemes…';
      const start = performance.now();
      const result = await align(samples, sampleRate, transcript);
      signal.throwIfAborted();
      alignMs = Math.round(performance.now() - start);
      duration = samples.length / sampleRate;
      waveform = makeWaveform(samples);
      words = result;
      audioUrl = URL.createObjectURL(wav);
      status = 'Alignment ready';
    } catch (failure) {
      if (!signal.aborted) {
        status = '';
        error = failure instanceof Error ? failure.message : String(failure);
      }
    } finally {
      busy = false;
    }
  }

  function makeWaveform(samples: Float32Array) {
    const columns = 420;
    return Array.from({ length: columns }, (_, column) => {
      let peak = 0;
      const end = Math.floor((column + 1) * samples.length / columns);
      for (let i = Math.floor(column * samples.length / columns); i < end; i++) {
        peak = Math.max(peak, Math.abs(samples[i]));
      }
      const height = Math.min(27, Math.max(0.5, peak * 40));
      return `M${column} ${28 - height}v${height * 2}`;
    }).join(' ');
  }

  function seek(start: number) {
    if (!audio) return;
    audio.currentTime = start;
    time = start;
  }

  const ms = (seconds: number) => Math.round(seconds * 1000);
</script>

<svelte:head>
  <title>TinyAligner · Phoneme timing lab</title>
  <meta name="description" content="Generate speech in your browser and inspect where each phoneme starts." />
</svelte:head>

<main>
  <header>
    <p class="eyebrow">MODEL DEMOS / TEXT TO PHONEMES</p>
    <h1>Phoneme timing lab</h1>
    <p>Generate English speech in your browser and see where each sound starts.</p>
  </header>

  <section class="panel">
    <form onsubmit={(event) => { event.preventDefault(); void generate(); }}>
      <label for="prompt">Text to speak</label>
      <textarea id="prompt" bind:value={text} maxlength="200" rows="2" disabled={busy}></textarea>
      <div class="actions">
        <button class="primary" disabled={busy || !text.trim()}>Generate &amp; align</button>
        <span role="status">{status}</span>
      </div>
    </form>
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    <p class="note">The first run downloads the speech and alignment models. Use short English sentences and spell out numbers.</p>
  </section>

  {#if audioUrl}
    <section class="panel results" aria-labelledby="results">
      <div class="heading">
        <h2 id="results">Alignment</h2>
        <span>{duration.toFixed(2)} s audio · {phones.length} phonemes · {alignMs} ms alignment</span>
      </div>
      <!-- svelte-ignore a11y_media_has_caption -->
      <audio aria-label="Generated speech" controls src={audioUrl} bind:this={audio} bind:currentTime={time}></audio>
      <div class="timeline">
        <svg viewBox="0 0 420 56" preserveAspectRatio="none" aria-hidden="true">
          <path d={waveform} fill="none" stroke="currentColor" stroke-width="0.75" />
        </svg>
        <div class="lane" aria-label="Phoneme intervals">
          {#each phones as phone}
            <span class:active={time >= phone.start && time < phone.end}
              style:left={`${100 * phone.start / duration}%`}
              style:width={`${100 * (phone.end - phone.start) / duration}%`}
              title={`${phone.word} · ${ms(phone.start)}–${ms(phone.end)} ms`}></span>
          {/each}
        </div>
        <div class="playhead" style:left={`${Math.min(100, 100 * time / duration)}%`}></div>
      </div>
      <div class="scale"><span>0.00 s</span><span>{duration.toFixed(2)} s</span></div>
      <p class="note">Select a word or phoneme to jump to its start. Phoneme labels use ARPABET. Times are CTC estimates on a 20 ms grid.</p>

      <div class="words">
        {#each words as word}
          <div class="word">
            <button class="word-title" type="button" onclick={() => seek(word.start)}>
              {word.word} <small>{ms(word.start)} ms</small>
            </button>
            <div class="phones">
              {#each word.phones as phone}
                <button type="button" class:active={time >= phone.start && time < phone.end}
                  onclick={() => seek(phone.start)}
                  title={`${ms(phone.start)}–${ms(phone.end)} ms`}>
                  {phone.word}<small>{ms(phone.start)} ms</small>
                </button>
              {/each}
            </div>
          </div>
        {/each}
      </div>
    </section>
  {/if}
</main>

<style>
  :global(*) { box-sizing: border-box; }
  :global(body) { margin: 0; background: #f4f5f1; color: #192821; font: 16px/1.5 system-ui, sans-serif; }
  main { max-width: 940px; margin: auto; padding: clamp(24px, 5vw, 64px) 20px; }
  header { margin-bottom: 26px; }
  h1 { margin: 0; font-size: clamp(2.15rem, 6vw, 3.7rem); line-height: 1.08; letter-spacing: -.05em; }
  h2, p { margin: 0; }
  header > p:last-child { margin-top: 12px; color: #485b50; }
  .eyebrow { color: #28634d; font: 700 .75rem/1.5 ui-monospace, monospace; letter-spacing: .13em; margin-bottom: 10px; }
  .panel { padding: clamp(16px, 4vw, 28px); background: white; border: 1px solid #d9e2d9; border-radius: 18px; margin-bottom: 18px; }
  label { display: block; font-weight: 700; margin-bottom: 8px; }
  textarea { width: 100%; resize: vertical; min-height: 74px; padding: 12px; font: inherit; color: inherit; background: #fafbf8; border: 1px solid #aab9ad; border-radius: 9px; }
  button { cursor: pointer; color: inherit; font: inherit; }
  button:disabled { opacity: .62; cursor: default; }
  button:focus-visible, textarea:focus-visible { outline: 2px solid #207952; outline-offset: 3px; }
  .actions, .heading, .scale { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }
  .actions { margin-top: 10px; }
  .heading, .scale { justify-content: space-between; }
  .heading span, .scale, .actions span, .note { color: #4a5d51; font-size: .86rem; }
  .primary { border: 0; background: #205b43; color: white; padding: 10px 16px; border-radius: 8px; font-weight: 700; }
  .note, .error { margin-top: 12px; }
  .error { color: #a33127; }
  audio { width: 100%; margin: 14px 0; }
  .timeline { position: relative; background: #eaf2ec; border-radius: 7px; overflow: hidden; }
  svg { width: 100%; height: 72px; color: #3c765c; display: block; }
  .lane { height: 15px; position: relative; }
  .lane span { position: absolute; height: 100%; border-right: 1px solid #fff; background: #94bca5; }
  .lane span.active { background: #255e47; }
  .playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: #203e31; pointer-events: none; }
  .scale { margin-top: 4px; font-variant-numeric: tabular-nums; }
  .words { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
  .word { padding: 9px; border: 1px solid #dce7de; border-radius: 9px; max-width: 100%; }
  .word-title { background: transparent; border: 0; padding: 1px 3px 7px; font-weight: 750; }
  small { color: #4a5d51; font: .69rem ui-monospace, monospace; }
  .phones { display: flex; flex-wrap: wrap; gap: 4px; }
  .phones button { display: grid; gap: 3px; border: 1px solid transparent; border-radius: 5px; background: #edf3ee; padding: 6px 7px; font: 700 .83rem ui-monospace, monospace; }
  .phones button:hover, .phones button.active { border-color: #3c765c; background: #d6e9dc; }
</style>

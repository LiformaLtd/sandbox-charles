<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { api, type Voice, type Health, type Speech } from '../lib/api';
  import { Recorder, normalizeFile } from '../lib/recorder';
  import '../app.css';

  let health = $state<Health>();
  let voices = $state<Voice[]>([]);
  let selectedId = $state('');
  const selected = $derived(voices.find(voice => voice.id === selectedId));
  let name = $state('My voice');
  let consent = $state(false);
  let recording = $state(false);
  let starting = $state(false);
  let seconds = $state(0);
  let level = $state(0);
  let clip = $state<Blob>();
  let clipUrl = $state('');
  let operation = $state('');
  let message = $state('');
  let error = $state('');
  let text = $state('The rain had stopped by morning. Across the river, a small café was already open, and the smell of fresh bread drifted down the street.');
  let result = $state<Speech>();
  let defaultResult = $state<Speech>();
  let resultText = $state('');
  let defaultText = $state('');
  let resultVoice = $state('');
  let outputAudio = $state<HTMLAudioElement>();
  let defaultAudio = $state<HTMLAudioElement>();
  let confirmingDelete = $state(false);
  let recorder: Recorder | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;
  let poll: ReturnType<typeof setInterval> | undefined;
  let disposed = false;
  const busy = $derived(Boolean(operation) || starting || recording);

  function fail(cause: unknown) {
    console.error('Voice lab operation failed', cause);
    error = cause instanceof Error ? cause.message : 'Something went wrong. Please try again.';
    message = '';
  }

  async function refresh() {
    try {
      const [nextHealth, nextVoices] = await Promise.all([api<Health>('health'), api<Voice[]>('voices')]);
      if (disposed) return;
      health = nextHealth; voices = nextVoices;
      if (!selectedId && voices.length) selectedId = voices[0].id;
      if (health.ready && poll) { clearInterval(poll); poll = undefined; }
    } catch (cause) {
      if (!disposed) { health = undefined; fail(cause); }
    }
  }

  onMount(() => { void refresh(); poll = setInterval(() => { void refresh(); }, 3000); });
  onDestroy(() => {
    disposed = true;
    clearInterval(timer); clearInterval(poll);
    void recorder?.dispose();
    if (clipUrl) URL.revokeObjectURL(clipUrl);
  });

  function useClip(next: Blob) {
    if (clipUrl) URL.revokeObjectURL(clipUrl);
    clip = next; clipUrl = URL.createObjectURL(next);
    message = 'Recording ready. Listen back, then create your voice.';
  }

  async function startRecording() {
    if (busy) return;
    error = ''; message = ''; starting = true; seconds = 0;
    try {
      recorder = new Recorder();
      await recorder.start(value => { if (!disposed) level = value; });
      if (disposed) { await recorder.dispose(); return; }
      recording = true;
      const started = performance.now();
      timer = setInterval(() => {
        seconds = Math.min(20, (performance.now() - started) / 1000);
        if (seconds >= 20) void stopRecording();
      }, 100);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'NotAllowedError') {
        fail(new Error('Microphone access was denied. Allow microphone access in your browser and try again.'));
      } else fail(cause);
    } finally { starting = false; }
  }

  async function stopRecording() {
    if (!recording || !recorder) return;
    clearInterval(timer); recording = false; operation = 'Preparing recording…'; level = 0;
    try { useClip(await recorder.stop()); } catch (cause) { fail(cause); }
    finally { recorder = undefined; operation = ''; }
  }

  async function upload(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    operation = 'Preparing recording…'; error = ''; message = '';
    try { useClip(await normalizeFile(file)); }
    catch (cause) { fail(cause); }
    finally { operation = ''; input.value = ''; }
  }

  async function clone() {
    if (!clip || !consent || busy) return;
    operation = 'Creating your voice…'; error = ''; message = '';
    try {
      const voice = await api<Voice>(`voices?name=${encodeURIComponent(name.trim())}`, {
        method: 'POST', headers: { 'Content-Type': 'audio/wav', 'X-Voice-Consent': 'own-voice' }, body: clip
      });
      voices = [voice, ...voices]; selectedId = voice.id; confirmingDelete = false;
      message = `“${voice.name}” is ready. Try it with some new text.`;
    } catch (cause) { fail(cause); }
    finally { operation = ''; }
  }

  async function generate(baseline = false) {
    if (busy || (!baseline && !selected) || !text.trim()) return;
    const voiceId = baseline ? 'default' : selected!.id;
    const label = baseline ? 'Default Phonon voice' : selected!.name;
    const spokenText = text.trim();
    operation = 'Generating speech…'; error = ''; message = ''; outputAudio?.pause(); defaultAudio?.pause();
    try {
      const speech = await api<Speech>('speech', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: spokenText, voiceId }) });
      if (baseline) { defaultResult = speech; defaultText = spokenText; }
      else { result = speech; resultText = spokenText; resultVoice = label; }
      message = `${label} is ready.`;
      await tick();
      // Browser autoplay policy can require an additional click; controls remain available.
      const player = baseline ? defaultAudio : outputAudio;
      if (player) void player.play().catch(() => { message = 'Speech ready. Press play to listen.'; });
    } catch (cause) { fail(cause); }
    finally { operation = ''; }
  }

  async function removeVoice() {
    if (!selected || busy) return;
    const id = selected.id;
    operation = 'Deleting voice…'; error = '';
    try {
      await api(`voices/${id}`, { method: 'DELETE' });
      voices = voices.filter(voice => voice.id !== id); selectedId = voices[0]?.id || '';
      if (result?.voiceId === id) { outputAudio?.pause(); result = undefined; }
      confirmingDelete = false; message = 'Voice and its saved recordings deleted.';
    } catch (cause) { fail(cause); }
    finally { operation = ''; }
  }
</script>

<svelte:head>
  <title>Phonon Voice Cloning Lab</title>
  <meta name="description" content="A local experiment in cloning your own voice with Phonon." />
</svelte:head>

<div class="shell">
  <header class="topbar">
    <a class="brand" href="/" aria-label="Phonon voice cloning home"><span class="brandmark">p</span> phonon <span class="brand-note">/ voice lab</span></a>
    <span class="local-pill"><span class:online={health?.ready} class="status-dot"></span>{health?.ready ? 'Running locally' : 'Connecting to local engine'}</span>
  </header>

  <main>
    <div class="intro">
      <p class="eyebrow">EXPERIMENT 02 · VOICE CLONING</p>
      <h1>A little recording.<br />A voice of your own.</h1>
      <p class="lede">Record yourself, create a voice, and hear it say something new.<br class="desktop-break" /> Your recordings and speech stay on this computer.</p>
    </div>

    {#if error}
      <div class="notice error" role="alert"><strong>Couldn’t finish that.</strong> {error}<button class="text-button" onclick={() => { error = ''; }}>Dismiss</button></div>
    {/if}
    {#if !health?.ready}
      <div class="notice" role="status">{health?.error || 'Starting the local voice engine…'} <button class="text-button" onclick={refresh}>Check again</button></div>
    {/if}
    <div class="live-status" role="status" aria-live="polite">{operation || message}</div>

    <div class="workspace">
      <section class="card capture-card" aria-labelledby="record-heading">
        <div class="step-heading"><span class="step">01</span><div><p class="eyebrow">THE REFERENCE</p><h2 id="record-heading">Capture your voice</h2></div></div>
        <p class="helper">Speak naturally in a quiet room for 10–15 seconds. Keep the microphone at a comfortable distance.</p>

        <div class="reading-prompt"><span class="small-label">SOMETHING TO READ</span><p>“I’m trying something a little different today. The window is open, the coffee is warm, and there’s plenty of time to see where this takes us.”</p></div>

        <div class:recording class="recorder">
          <div class="mic-orbit" style={`--level:${level}`}><span class="mic-symbol">{recording ? '●' : '↟'}</span></div>
          <p class="record-state">{recording ? 'Listening to you' : starting ? 'Opening microphone…' : clip ? 'Recording captured' : 'Ready when you are'}</p>
          <div class="timer">{seconds.toFixed(1)}<span> / 20 sec</span></div>
          <div class="level-track" aria-hidden="true"><span style={`width:${level * 100}%`}></span></div>
          {#if recording}
            <button class="primary recording-button" onclick={stopRecording}><span>■</span> Stop recording</button>
          {:else}
            <button class="primary" onclick={startRecording} disabled={busy || !health?.ready}>{clip ? 'Record again' : 'Record my voice'}</button>
          {/if}
          <label class="upload-button" class:disabled={busy || !health?.ready}>or use a recording<input aria-label="Use a voice recording" type="file" accept="audio/wav,audio/x-wav,.wav" onchange={upload} disabled={busy || !health?.ready} /></label>
        </div>

        {#if clipUrl}
          <div class="recording-preview"><label for="reference-preview" class="small-label">YOUR RECORDING</label><audio id="reference-preview" aria-label="Your recorded voice" src={clipUrl} controls></audio></div>
        {/if}

        <div class="create-form">
          <label for="voice-name">Voice name</label>
          <input id="voice-name" bind:value={name} maxlength="60" placeholder="My voice" disabled={busy} />
          <label class="consent"><input type="checkbox" bind:checked={consent} disabled={busy} /><span>This is my own voice, and I want to create a synthetic version of it.</span></label>
          <button class="primary full" onclick={clone} disabled={!clip || !consent || !name.trim() || busy || !health?.ready}>{operation === 'Creating your voice…' ? 'Creating your voice…' : 'Create my voice'}<span aria-hidden="true">↗</span></button>
          <p class="fine-print">Uses up to 10 seconds of speech. No model retraining required.</p>
        </div>
      </section>

      <section class="card speech-card" aria-labelledby="speech-heading">
        <div class="step-heading"><span class="step">02</span><div><p class="eyebrow">THE NEW WORDS</p><h2 id="speech-heading">Hear yourself differently</h2></div></div>
        <p class="helper">Choose your saved voice and give it something to say. Phonon 31M speaks English.</p>

        {#if voices.length}
          <label for="voice-select">Your voices <span class="count">{voices.length}</span></label>
          <select id="voice-select" bind:value={selectedId} disabled={busy} onchange={() => { confirmingDelete = false; }}>{#each voices as voice}<option value={voice.id}>{voice.name}</option>{/each}</select>
          {#if selected}
            <div class="voice-detail"><span>{selected.usedSeconds.toFixed(1)}s reference · created in {(selected.creationMs / 1000).toFixed(1)}s</span><button class="text-button danger" onclick={() => { confirmingDelete = !confirmingDelete; }} disabled={busy}>Delete voice</button></div>
            {#if confirmingDelete}<div class="delete-confirm">Delete this voice, its reference and its generated speech?<div><button class="danger-button" onclick={removeVoice} disabled={busy}>Delete permanently</button><button class="text-button" onclick={() => { confirmingDelete = false; }}>Keep voice</button></div></div>{/if}
            <details class="reference-details"><summary>Listen to the saved reference</summary><audio aria-label="Saved voice reference" src={`/api/voices/${selected.id}/reference`} controls></audio><a href={`/api/voices/${selected.id}/embedding`} download>Download voice file</a></details>
          {/if}
        {:else}
          <div class="empty-voice"><span class="empty-wave">ııııııı</span><strong>Your voice goes here</strong><p>Create a voice from your recording to get started.</p></div>
        {/if}

        <label for="speech-text" class="text-label">What would you like to say?</label>
        <textarea id="speech-text" bind:value={text} maxlength="1000" rows="7" disabled={busy}></textarea>
        <div class="text-meta"><span>Try a sentence you didn’t record.</span><span>{text.length}/1,000</span></div>
        <button class="primary full" onclick={() => generate()} disabled={busy || !selected || !text.trim() || !health?.ready}>{operation === 'Generating speech…' ? 'Generating speech…' : 'Generate speech'}<span aria-hidden="true">▷</span></button>
        <button class="secondary full" onclick={() => generate(true)} disabled={busy || !text.trim() || !health?.ready}>Generate default voice</button>
        <p class="fine-print">The default is Phonon’s original voice. Both clips stay available below.</p>

        {#if result}
          <div class="result" data-testid="speech-result"><div class="result-heading"><span class="small-label">CLONED VOICE</span><span class="synthetic-tag">Synthetic</span></div><h3>{resultVoice}</h3><audio bind:this={outputAudio} aria-label="Generated speech" src={result.url} controls onplay={() => defaultAudio?.pause()}></audio><div class="result-meta"><span>{result.durationSeconds.toFixed(1)}s audio · {(result.generationMs / 1000).toFixed(2)}s generation</span><a href={result.url} download="phonon-cloned-voice.wav">Download WAV ↓</a></div><details class="spoken-text"><summary>Text used</summary><p>{resultText}</p></details></div>
        {/if}
        {#if defaultResult}
          <div class="result baseline-result" data-testid="default-result"><div class="result-heading"><span class="small-label">DEFAULT VOICE · COMPARISON</span><span class="synthetic-tag">Synthetic</span></div><h3>Default Phonon voice</h3><audio bind:this={defaultAudio} aria-label="Default voice speech" src={defaultResult.url} controls onplay={() => outputAudio?.pause()}></audio><div class="result-meta"><span>{defaultResult.durationSeconds.toFixed(1)}s audio · {(defaultResult.generationMs / 1000).toFixed(2)}s generation</span><a href={defaultResult.url} download="phonon-default-voice.wav">Download WAV ↓</a></div><details class="spoken-text"><summary>Text used</summary><p>{defaultText}</p></details></div>
          {#if result && resultText !== defaultText}<p class="comparison-warning">These clips use different text. Generate both again with the same sentence for a fair comparison.</p>{/if}
        {/if}
      </section>
    </div>

    <aside class="experiment-note"><span class="note-icon">✳</span><div><strong>A listening experiment, not a perfect replica.</strong><p>The community encoder can capture some vocal characteristics, but similarity varies. Compare the source and generated clips with your own ears.</p></div><a href="https://github.com/connerohnesorge/phonon-encoder" target="_blank" rel="noreferrer">About the encoder ↗</a></aside>
  </main>
  <footer><span>PHONON VOICE CLONING LAB</span><span>Community speaker encoder · Phonon 31M · Local CPU</span></footer>
</div>

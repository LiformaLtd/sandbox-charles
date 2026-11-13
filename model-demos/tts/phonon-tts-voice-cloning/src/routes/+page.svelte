<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { api, type Voice, type Health, type Speech } from '../lib/api';
  import { Recorder, normalizeFile } from '../lib/recorder';
  import { recordingFor, methodName, versionName, finalError, formatError } from '../lib/voice-workflow';
  import SharedComparison from '../lib/SharedComparison.svelte';
  import '../app.css';

  let health = $state<Health>();
  let voices = $state<Voice[]>([]);
  let selectedId = $state('');
  const selected = $derived(voices.find(voice => voice.id === selectedId));
  const recordingSource = $derived(selected ? recordingFor(selected, voices) : undefined);
  const recordings = $derived([...new Map(voices.map(voice => {
    const source = recordingFor(voice, voices); return [source.id, source] as const;
  })).values()]);
  const family = $derived(recordingSource ? voices.filter(voice => recordingFor(voice, voices).id === recordingSource.id)
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt)) : []);
  const selectedVersion = $derived(selected ? versionName(selected, family) : '');
  const parentVersion = $derived(selected?.parentId ? voices.find(voice => voice.id === selected.parentId) : undefined);
  const hasChildren = $derived(voices.some(voice => voice.parentId === selectedId));
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
  let transcript = $state('');
  let transcriptConfirmed = $state(false);
  let fitIterations = $state(20);
  let fitMethod = $state<'iterative' | 'linear'>('linear');
  let fitFocus = $state<'voice' | 'pronunciation'>('voice');
  let fitting = $state(false);
  let captureOpen = $state(false);
  let selectionNote = $state('');
  let cancelling = $state(false);
  let beforeFitAudio = $state<HTMLAudioElement>();
  let afterFitAudio = $state<HTMLAudioElement>();
  let referenceAudio = $state<HTMLAudioElement>();
  let captureAudio = $state<HTMLAudioElement>();
  let recorder: Recorder | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;
  let poll: ReturnType<typeof setInterval> | undefined;
  let disposed = false;
  let refreshVersion = 0;
  let loadedSelection = false;
  const busy = $derived(Boolean(operation) || starting || recording || Boolean(health?.busy));

  function pausePlayers(except?: HTMLAudioElement) {
    for (const player of [outputAudio, defaultAudio, beforeFitAudio, afterFitAudio, referenceAudio, captureAudio]) {
      if (player && player !== except) player.pause();
    }
  }

  function fail(cause: unknown) {
    console.error('Voice lab operation failed', cause);
    error = cause instanceof Error ? cause.message : 'Something went wrong. Please try again.';
    message = '';
  }

  async function refresh() {
    const version = ++refreshVersion;
    try {
      const [nextHealth, nextVoices] = await Promise.all([api<Health>('health'), api<Voice[]>('voices')]);
      if (disposed || version !== refreshVersion) return;
      health = nextHealth; voices = nextVoices;
      if (!loadedSelection || !voices.some(voice => voice.id === selectedId)) {
        const existing = voices.find(voice => voice.id === selectedId);
        selectVersion(existing?.id || (voices.length ? recordingFor(voices[0], voices).id : ''));
        if (!loadedSelection) captureOpen = !voices.length;
        loadedSelection = true;
      }
    } catch (cause) {
      if (!disposed && version === refreshVersion) { health = undefined; fail(cause); }
    }
  }

  onMount(() => {
    try { selectedId = localStorage.getItem('phonon-voice-lab:selected-version') || ''; }
    catch { selectionNote = 'This browser cannot remember your selected version. Saved voices are still kept on this computer.'; }
    void refresh(); poll = setInterval(() => { void refresh(); }, 2000);
  });
  onDestroy(() => {
    disposed = true;
    clearInterval(timer); clearInterval(poll);
    void recorder?.dispose();
    if (clipUrl) URL.revokeObjectURL(clipUrl);
  });

  function useClip(next: Blob) {
    if (clipUrl) URL.revokeObjectURL(clipUrl);
    clip = next; clipUrl = URL.createObjectURL(next);
    message = 'New recording captured. Save it below to create its initial voice.';
  }

  async function startRecording() {
    if (busy) return;
    error = ''; message = ''; starting = true; seconds = 0;
    pausePlayers();
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
      ++refreshVersion;
      voices = [voice, ...voices]; selectVersion(voice.id); captureOpen = false;
      clip = undefined; if (clipUrl) URL.revokeObjectURL(clipUrl); clipUrl = ''; consent = false; seconds = 0;
      message = `Saved “${voice.name}” and selected its initial clone. You can speak with it now or create a fitted version.`;
    } catch (cause) { fail(cause); }
    finally { operation = ''; }
  }

  async function generate(baseline = false) {
    if (busy || (!baseline && !selected) || !text.trim()) return;
    const voiceId = baseline ? 'default' : selected!.id;
    const label = baseline ? 'Default Phonon voice' : `${recordingSource!.name} · ${selectedVersion}`;
    const spokenText = text.trim();
    operation = 'Generating speech…'; error = ''; message = ''; pausePlayers();
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
      ++refreshVersion;
      voices = voices.filter(voice => voice.id !== id);
      selectVersion(voices.length ? recordingFor(voices[0], voices).id : '');
      if (!voices.length) captureOpen = true;
      if (result?.voiceId === id) { outputAudio?.pause(); result = undefined; }
      confirmingDelete = false; message = 'Voice version deleted.';
    } catch (cause) { fail(cause); }
    finally { operation = ''; }
  }

  function selectVersion(id: string) {
    selectedId = id;
    confirmingDelete = false;
    transcript = voices.find(voice => voice.id === id)?.transcript || '';
    transcriptConfirmed = false;
    try { localStorage.setItem('phonon-voice-lab:selected-version', id); }
    catch { selectionNote = 'This browser cannot remember your selected version. Saved voices are still kept on this computer.'; }
  }

  async function transcribe() {
    if (!selected || busy) return;
    operation = 'Transcribing your saved recording locally…'; error = ''; message = '';
    try {
      const result = await api<{ text: string }>(`voices/${selected.id}/transcribe`, { method: 'POST' });
      transcript = result.text; transcriptConfirmed = false;
      message = 'Check these words against your recording and correct any mistakes.';
    } catch (cause) { fail(cause); }
    finally { operation = ''; await refresh(); }
  }

  async function refine() {
    if (!selected || busy || (fitMethod === 'iterative' && (!transcriptConfirmed || !transcript.trim())) || (fitMethod === 'linear' && !health?.linearReady)) return;
    const id = selected.id;
    fitting = true; operation = 'Fitting your voice…'; error = ''; message = ''; pausePlayers();
    try {
      const candidate = await api<Voice | { cancelled: true }>(`voices/${id}/refine`, { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: transcript.trim(), confirmed: transcriptConfirmed, iterations: fitIterations, focus: fitFocus, method: fitMethod }) });
      if ('cancelled' in candidate) message = 'Voice fitting cancelled. Your original voice is unchanged.';
      else {
        ++refreshVersion;
        voices = [candidate, ...voices]; selectVersion(candidate.id);
        message = 'New fitted version saved and selected. Listen to the before-and-after clips in step 3. Earlier versions are kept.';
      }
    } catch (cause) { fail(cause); }
    finally { fitting = false; cancelling = false; operation = ''; await refresh(); }
  }

  async function cancelFit() {
    if (!health?.refinement || cancelling) return;
    cancelling = true;
    try { await api(`refinements/${health.refinement.id}/cancel`, { method: 'POST' }); }
    catch (cause) { fail(cause); cancelling = false; }
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
      <p class="lede">Choose a saved recording, compare voice versions, and hear them speak.<br class="desktop-break" /> Your recordings and speech stay on this computer.</p>
    </div>

    {#if error}
      <div class="notice error" role="alert"><strong>Couldn’t finish that.</strong> {error}<button class="text-button" onclick={() => { error = ''; }}>Dismiss</button></div>
    {/if}
    {#if !health?.ready}
      <div class="notice" role="status">{health?.error || 'Starting the local voice engine…'} <button class="text-button" onclick={refresh}>Check again</button></div>
    {/if}
    <div class="live-status" role="status" aria-live="polite">{operation || message}</div>

    <div class="workflow-guide"><strong>Recording → Voice version → Speech</strong><p>Your recording and voice versions are saved. Reloading restores your last selected version. New fitting starts only when you press its create button.</p></div>
    {#if selectionNote}<p class="helper">{selectionNote}</p>{/if}
    <div class="workspace">
      <section class="card library-card" aria-labelledby="record-heading">
        <div class="step-heading"><span class="step">01</span><div><p class="eyebrow">THE SAVED AUDIO</p><h2 id="record-heading">Choose a recording</h2></div></div>
        {#if recordingSource}
          <p class="helper">Use a recording already saved here, or record a new sample below.</p>
          <div class="source-grid">
            <div><label for="recording-select">Saved recording</label><select id="recording-select" value={recordingSource.id} disabled={busy} onchange={event => selectVersion(event.currentTarget.value)}>{#each recordings as source (source.id)}<option value={source.id}>{source.name} · {source.recordingSeconds.toFixed(1)}s</option>{/each}</select><p class="fine-print">Saved {new Date(recordingSource.createdAt).toLocaleString()} · reused by the fitting in section 2 and voice versions in section 3.</p></div>
            <div><label for="saved-reference">Your actual recorded voice</label><audio id="saved-reference" bind:this={referenceAudio} aria-label="Saved voice reference" src={`/api/voices/${selectedId}/reference`} controls onplay={event => pausePlayers(event.currentTarget)}></audio><p class="fine-print">Listen here to check which recording will be used for fitting.</p></div>
          </div>
        {:else}<p class="helper">Start by recording a short sample. You only need to do this once for each recording you want to keep.</p>{/if}
        <details class="new-recording" bind:open={captureOpen}>
          <summary>Record a new sample</summary>
          <p class="helper">This creates a separate saved recording. Your existing recordings and versions are kept.</p>
        <p class="helper">Speak naturally in a quiet room for 10–15 seconds. Keep the microphone at a comfortable distance.</p>

        <div class="reading-prompt"><span class="small-label">SOMETHING TO READ · OPTIONAL</span><p>“I’m trying something a little different today. The window is open, the coffee is warm, and there’s plenty of time to see where this takes us.”</p><small>Or say anything naturally. Your recording does not need to match the text in the speech box.</small></div>

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
          <p class="notice">New recording, not saved yet. Save it below before fitting or generating speech with it.</p>
          <div class="recording-preview"><label for="reference-preview" class="small-label">YOUR RECORDING</label><audio id="reference-preview" bind:this={captureAudio} onplay={() => pausePlayers(captureAudio)} aria-label="Your recorded voice" src={clipUrl} controls></audio></div>
        {/if}

        <div class="create-form">
          <label for="voice-name">Recording name</label>
          <input id="voice-name" bind:value={name} maxlength="60" placeholder="My voice" disabled={busy} />
          <label class="consent"><input type="checkbox" bind:checked={consent} disabled={busy} /><span>This is my own voice, and I want to create a synthetic version of it.</span></label>
          <button class="primary full" onclick={clone} disabled={!clip || !consent || !name.trim() || busy || !health?.ready}>{operation === 'Creating your voice…' ? 'Creating your voice…' : 'Save recording & create initial clone'}<span aria-hidden="true">↗</span></button>
          <p class="fine-print">Saving keeps this recording and creates its initial clone. Fitted versions are optional.</p>
        </div>

        </details>
      </section>

      <section class="card versions-card" aria-labelledby="versions-heading">
        <div class="step-heading"><span class="step">02</span><div><p class="eyebrow">MAKE A NEW VERSION · OPTIONAL</p><h2 id="versions-heading">Create a fitted version</h2></div></div>
        {#if selected && recordingSource}
          <p class="helper">Create a new fitted version from the saved recording “{recordingSource.name}”. This is optional: you can play any existing version in section 3.</p>
          <div class="fitting-panel">
            <h3>Choose a fitting method</h3>
            <p>Choose a method below and press its create button. The new version is saved separately and selected when it finishes.</p>
              {#if health?.refinementReady}
                <label for="fit-method">New version method</label>
                <select id="fit-method" bind:value={fitMethod} disabled={busy}>
                  <option value="iterative">Iterative fit · about 1–5 minutes</option>
                  <option value="linear" disabled={!health.supportsLinear}>Fast linear fit · usually seconds</option>
                </select>
                {#if fitMethod === 'linear'}
                  <p>Uses the saved recording “{recordingSource.name}” to make a new voice estimate. You do not need to record again or type a transcript.</p>
                  <p class="fine-print">The selected version stays available as the “before” comparison. A linear fit may sound better or worse.</p>
                  {#if !health.linearReady}<p>Fast fitting data is not installed yet. Run npm run setup:linear in this lab, then refresh.</p>{/if}
                {:else}
                <p>Starts from {selectedVersion}, currently selected in section 3, and uses the saved recording “{recordingSource.name}”. Check the words spoken in that recording before fitting.</p>
                <button class="secondary full" onclick={transcribe} disabled={busy}>Get transcript from saved recording</button>
                <label for="reference-transcript">The actual words in your recording</label>
                <textarea id="reference-transcript" rows="3" maxlength="1000" bind:value={transcript} oninput={() => { transcriptConfirmed = false; }} disabled={busy} placeholder="Transcribe locally, or type exactly what you said."></textarea>
                <label class="consent"><input type="checkbox" bind:checked={transcriptConfirmed} disabled={busy} /><span>I checked this transcript against my recording and corrected any mistakes.</span></label>
                <label for="fit-focus">Fitting focus</label><select id="fit-focus" bind:value={fitFocus} disabled={busy}><option value="voice">Voice similarity</option><option value="pronunciation" disabled={!health.supportsPronunciation}>Voice and pronunciation · experimental</option></select>
                {#if fitFocus === 'pronunciation'}<p>This gives recorded pronunciation more weight. Acoustic matching can improve, but accent and voice likeness still need listening checks.</p>{/if}
                <label for="fit-duration">Fitting time</label><select id="fit-duration" bind:value={fitIterations} disabled={busy}><option value={20}>Quick · about 1–2 minutes</option><option value={60}>Longer · about 3–5 minutes</option></select>
                {/if}
                <button class="secondary full" onclick={refine} disabled={busy || (fitMethod === 'linear' ? !health.linearReady : !transcriptConfirmed || transcript.trim().length < 3)}>{fitMethod === 'linear' ? 'Create linear-fit version' : 'Create iterative-fit version'}</button>
                {#if health.refinement}
                  <div class="fit-progress" role="status"><strong>{health.refinement.stage}</strong><progress value={health.refinement.step} max={health.refinement.total}></progress><span>{health.refinement.step} / {health.refinement.total} steps</span><button class="text-button" onclick={cancelFit} disabled={cancelling}>{cancelling ? 'Cancelling…' : 'Cancel fitting'}</button></div>
                {/if}
              {:else if health?.refinementReady === false}<p>Local fitting models are not installed. Setup instructions are in this lab’s README.</p>
              {:else if health?.ready}<p>Restart this lab to enable the new fitting controls.</p>
              {:else}<p>Waiting for the local voice engine…</p>{/if}

          </div>

        {:else}<div class="empty-voice"><strong>No saved versions yet</strong><p>Save a recording in step 1 to create its initial clone.</p></div>{/if}
      </section>

      <section class="card speech-card" aria-labelledby="speech-heading">
        <div class="step-heading"><span class="step">03</span><div><p class="eyebrow">CHOOSE WHAT TO PLAY</p><h2 id="speech-heading">Choose a voice and listen</h2></div></div>
        {#if selected && recordingSource}
          <p class="helper">These versions all use the recording “{recordingSource.name}”. Selecting one makes it ready to use; it does not run fitting.</p>
          <label for="voice-select">Voice version</label>
          <select id="voice-select" value={selectedId} disabled={busy} onchange={event => selectVersion(event.currentTarget.value)}>{#each family as version (version.id)}<option value={version.id}>{versionName(version, family)} · {new Date(version.createdAt).toLocaleString()}</option>{/each}</select>
          <div class="selected-version"><span class="small-label">SELECTED FOR SPEECH</span><strong>{selectedVersion}</strong><span>{selected.fit ? `Saved fit · ${selected.fit.elapsedSeconds.toFixed(2)}s to create` : 'The first clone made when this recording was saved.'}</span>{#if selected.fit}<strong class="fit-error" data-testid="fit-error">{selected.fit.algorithm === 'linear' ? 'Linear fitting error' : 'Iterative fitting error'}: {formatError(finalError(selected))}</strong><small>{selected.fit.algorithm === 'linear' ? 'Linear fit objective' : 'Iterative fitting objective'}. Lower is better within the same method and settings; errors from different methods are not directly comparable.</small>{/if}</div>

          <details class="version-management"><summary>Manage this saved version</summary><p class="fine-print">Saved name: {selected.name}</p><a href={`/api/voices/${selected.id}/embedding`} download>Download voice file</a>
            <button class="text-button danger" onclick={() => { confirmingDelete = !confirmingDelete; }} disabled={busy || hasChildren}>Delete version</button>
            {#if hasChildren}<p class="fine-print">Other versions were made from this one. Delete those versions first to keep the recording’s history together.</p>{/if}
            {#if confirmingDelete}<div class="delete-confirm">Delete this version and its generated speech?<div><button class="danger-button" onclick={removeVoice} disabled={busy}>Delete permanently</button><button class="text-button" onclick={() => { confirmingDelete = false; }}>Keep version</button></div></div>{/if}
          </details>
          <SharedComparison source={recordingSource} versions={family} {selectedId} {health} disabled={busy}
            onactivity={value => { operation = value; }} onrefresh={refresh} onerror={fail} />
          <p class="helper">Using <strong>{selectedVersion}</strong> from “{recordingSource.name}”.</p>
          {#if selected.fit}
            <div class="fit-comparison">
              <h3>Saved before-and-after comparison</h3>
              <p class="helper">These clips were saved when this version was created. They use the same sentence and speech settings.</p>
              <p class="comparison-sentence">“{selected.fit.heldOutText}”</p>
              <label for="before-fit">Before: {parentVersion ? versionName(parentVersion, family) : 'Previous version'}</label><audio id="before-fit" bind:this={beforeFitAudio} aria-label="Before fit comparison" src={`/api/voices/${selected.id}/original-comparison`} controls onplay={() => pausePlayers(beforeFitAudio)}></audio>
              <label for="after-fit">After: {selectedVersion}</label><audio id="after-fit" bind:this={afterFitAudio} aria-label="After fit comparison" src={`/api/voices/${selected.id}/fitted-comparison`} controls onplay={() => pausePlayers(afterFitAudio)}></audio>
              <p class="fine-print">The text box below creates new speech. It does not change these saved comparison clips.</p>
              {#if !selected.fit.generalized}<p class="comparison-warning">The automated similarity check did not improve. Listen before choosing this version.</p>{/if}
              <details class="fit-measurements"><summary>Fit measurements</summary>
                <p class="fine-print">{methodName(selected)} · total {selected.fit.elapsedSeconds.toFixed(2)}s.{#if selected.fit.algorithm === 'linear'} {selected.fit.components} directions · solve {selected.fit.solveMs?.toFixed(1)}ms.{/if}</p>
                <p class="fine-print">Speaker comparison score: {selected.fit.heldOutBefore.speakerCosine.toFixed(2)} → {selected.fit.heldOutAfter.speakerCosine.toFixed(2)}. This is a diagnostic score, not an accent score or percentage match.</p>
              </details>
            </div>
          {:else}<p class="helper">This is the initial clone. You can generate speech now, or optionally make a fitted version in step 2.</p>{/if}
        <label for="speech-text" class="text-label">Text to generate</label>
        <textarea id="speech-text" bind:value={text} maxlength="1000" rows="7" disabled={busy}></textarea>
        <div class="text-meta"><span>Try a sentence you didn’t record.</span><span>{text.length}/1,000</span></div>
        <button class="primary full" onclick={() => generate()} disabled={busy || !selected || !text.trim() || !health?.ready}>{operation === 'Generating speech…' ? 'Generating speech…' : 'Generate with selected version'}<span aria-hidden="true">▷</span></button>
        <button class="secondary full" onclick={() => generate(true)} disabled={busy || !text.trim() || !health?.ready}>Generate default comparison</button>
        <p class="fine-print">Generating reads the text above with {selectedVersion}. It does not run fitting. The default comparison uses Phonon’s preset voice.</p>

        {#if result}
          <div class="result" data-testid="speech-result"><div class="result-heading"><span class="small-label">CLONED VOICE</span><span class="synthetic-tag">Synthetic</span></div><h3>{resultVoice}</h3><audio bind:this={outputAudio} aria-label="Generated speech" src={result.url} controls onplay={() => pausePlayers(outputAudio)}></audio><div class="result-meta"><span>{result.durationSeconds.toFixed(1)}s audio · {(result.generationMs / 1000).toFixed(2)}s generation</span><a href={result.url} download="phonon-cloned-voice.wav">Download WAV ↓</a></div><details class="spoken-text"><summary>Text used</summary><p>{resultText}</p></details></div>
        {/if}
        {#if defaultResult}
          <div class="result baseline-result" data-testid="default-result"><div class="result-heading"><span class="small-label">DEFAULT VOICE · COMPARISON</span><span class="synthetic-tag">Synthetic</span></div><h3>Default Phonon voice</h3><audio bind:this={defaultAudio} aria-label="Default voice speech" src={defaultResult.url} controls onplay={() => pausePlayers(defaultAudio)}></audio><div class="result-meta"><span>{defaultResult.durationSeconds.toFixed(1)}s audio · {(defaultResult.generationMs / 1000).toFixed(2)}s generation</span><a href={defaultResult.url} download="phonon-default-voice.wav">Download WAV ↓</a></div><details class="spoken-text"><summary>Text used</summary><p>{defaultText}</p></details></div>
          {#if result && resultText !== defaultText}<p class="comparison-warning">These clips use different text. Generate both again with the same sentence for a fair comparison.</p>{/if}
        {/if}

        {:else}<div class="empty-voice"><strong>Choose a voice first</strong><p>Your saved voice versions will be ready to speak here.</p></div>{/if}
      </section>
    </div>

    <aside class="experiment-note"><span class="note-icon">✳</span><div><strong>A listening experiment, not a perfect replica.</strong><p>The community encoder can capture some vocal characteristics, but similarity varies. Compare the source and generated clips with your own ears.</p></div><a href="https://github.com/connerohnesorge/phonon-encoder" target="_blank" rel="noreferrer">About the encoder ↗</a></aside>
  </main>
  <footer><span>PHONON VOICE CLONING LAB</span><span>Community speaker encoder · Phonon 31M · Local CPU</span></footer>
</div>

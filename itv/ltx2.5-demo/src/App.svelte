<script lang="ts">
  import { onMount } from 'svelte';
  import { ACTIVE, OUTPUT, read, write, forget } from './storage';
  import type { Draft, Job, Output } from './storage';

  let draft: Draft = $state({ first: null, last: null, prompt: '', size: '768x512', frames: 121, seed: -1 });
  let firstUrl = $state('');
  let lastUrl = $state('');
  let videoUrl = $state('');
  let outputId = $state('');
  let outputCached = $state(false);
  let job: Job | null = $state(null);
  let ready = $state(false);
  let modelReady = $state(false);
  let connected = $state(false);
  let modelDetail = $state('Connecting to your local model server…');
  let error = $state('');
  let saved = $state('Restoring your workspace…');
  let log = $state('');
  function isBusy(value: Job | null) { return !!value && ['submitting', 'queued', 'running'].includes(value.status); }
  let busy = $derived(isBusy(job));
  let alive = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let healthTimer: ReturnType<typeof setTimeout> | undefined;
  const controller = new AbortController();

  function describe(cause: unknown) { return cause instanceof Error ? cause.message : String(cause); }
  async function api(path: string, options?: RequestInit) {
    const response = await fetch(`/api${path}`, { ...options, signal: controller.signal });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(typeof body?.detail === 'string' ? body.detail : `Local server returned ${response.status}`);
    }
    return response;
  }
  function replacePreview(which: 'first' | 'last') {
    const old = which === 'first' ? firstUrl : lastUrl;
    if (old) URL.revokeObjectURL(old);
    const file = draft[which];
    const url = file ? URL.createObjectURL(file) : '';
    if (which === 'first') firstUrl = url; else lastUrl = url;
  }
  async function saveDraft() {
    saved = 'Saving…';
    try {
      await write('draft', $state.snapshot(draft));
      saved = 'Saved on this browser';
    } catch (cause) {
      saved = 'Changes could not be saved';
      error = `Browser storage: ${describe(cause)}. Free browser storage and try again.`;
      console.error('Could not persist LTX draft', cause);
    }
  }
  async function choose(event: Event, which: 'first' | 'last') {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 20 * 1024 * 1024) {
      error = 'Choose a PNG, JPEG or WebP image under 20MB.';
      input.value = '';
      return;
    }
    draft[which] = file;
    replacePreview(which);
    await saveDraft();
    input.value = '';
  }
  function rememberJob(value: Job) {
    localStorage.setItem(ACTIVE, JSON.stringify(value));
    job = value;
  }
  function showOutput(output: Output, cached = false) {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    videoUrl = URL.createObjectURL(output.blob);
    outputId = output.id;
    outputCached = cached;
  }
  async function health() {
    try {
      const value = await (await api('/health')).json();
      connected = true;
      modelReady = value.model_ready;
      modelDetail = value.detail;
    } catch (cause) {
      if (!alive) return;
      connected = false;
      modelReady = false;
      modelDetail = 'Local model server is offline. Your saved workspace is still here.';
      console.error('LTX health check failed', cause);
    }
    if (alive) healthTimer = setTimeout(health, 10000);
  }
  async function postSaved(id: string) {
    const payload = await read<Draft>(`submission:${id}`);
    if (!payload?.first) throw new Error('Saved submission is missing. Select your image and submit a new job.');
    const form = new FormData();
    form.set('job_id', id);
    form.set('first', payload.first, payload.first.name);
    if (payload.last) form.set('last', payload.last, payload.last.name);
    form.set('prompt', payload.prompt);
    const [width, height] = payload.size.split('x');
    form.set('width', width); form.set('height', height);
    form.set('frames', String(payload.frames)); form.set('seed', String(payload.seed));
    const response = await fetch('/api/v1/jobs', { method: 'POST', body: form, signal: controller.signal });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      if (response.status >= 400 && response.status < 600) {
        rememberJob({ id, status: 'failed', error: detail?.detail || `Submission failed (${response.status})` });
      }
      throw new Error(typeof detail?.detail === 'string' ? detail.detail : `Submission failed (${response.status})`);
    }
    rememberJob(await response.json());
  }
  async function poll() {
    if (!job || !alive) return;
    const id = job.id;
    try {
      // A page may disappear between sending POST and receiving its response.
      // Reconcile by UUID, then resubmit the same durable snapshot if absent.
      const response = await fetch(`/api/v1/jobs/${id}`, { signal: controller.signal });
      if (response.status === 404 && job.status === 'submitting') {
        await postSaved(id);
      } else if (response.ok) {
        rememberJob(await response.json());
      } else {
        throw new Error(`Could not reconnect to job (${response.status})`);
      }
      if (job?.status === 'completed' && outputId !== id) {
        const blob = await (await api(`/v1/jobs/${id}/video`)).blob();
        const output = { id, blob };
        showOutput(output);
        await write('latestOutput', output);
        localStorage.setItem(OUTPUT, JSON.stringify({ id, bytes: blob.size }));
        outputCached = true;
        await forget(`submission:${id}`);
        saved = 'Video and workspace saved on this browser';
      }
      if (job?.status === 'failed') error = job.error || 'Generation failed. Open the generation log for details.';
      if (job?.status === 'running' || job?.status === 'failed') {
        log = await (await api(`/v1/jobs/${id}/log`)).text();
      }
    } catch (cause) {
      if (!alive) return;
      error = `Job recovery: ${describe(cause)}. Your job remains on the local server.`;
      console.error('LTX job recovery failed', cause);
    }
    if (alive && job && (['submitting', 'queued', 'running'].includes(job.status) ||
        (job.status === 'completed' && outputId !== job.id))) timer = setTimeout(poll, 2000);
  }
  async function generate(event: SubmitEvent) {
    event.preventDefault();
    if (!draft.first || !draft.prompt.trim() || busy) return;
    error = ''; log = '';
    const id = crypto.randomUUID();
    try {
      await write('draft', $state.snapshot(draft));
      await write(`submission:${id}`, $state.snapshot(draft));
      rememberJob({ id, status: 'submitting' });
      await navigator.storage?.persist?.();
      await postSaved(id);
    } catch (cause) {
      error = describe(cause);
      console.error('LTX submission failed', cause);
    }
    if (job?.id === id && job.status !== 'failed') await poll();
  }
  onMount(() => {
    alive = true;
    void (async () => {
      try {
        const stored = await read<Draft>('draft');
        if (stored) draft = stored;
        replacePreview('first'); replacePreview('last');
        const output = await read<Output>('latestOutput');
        if (output) showOutput(output, true);
        const active = localStorage.getItem(ACTIVE);
        if (active) job = JSON.parse(active);
        saved = output ? 'Video and workspace restored' : 'Workspace saved on this browser';
        ready = true;
        await health();
        if (job && job.status !== 'failed') await poll();
      } catch (cause) {
        error = `Could not restore browser storage: ${describe(cause)}`;
        console.error('Could not restore LTX workspace', cause);
      }
    })();
    return () => {
      alive = false;
      clearTimeout(timer); clearTimeout(healthTimer);
      controller.abort();
      for (const url of [firstUrl, lastUrl, videoUrl]) if (url) URL.revokeObjectURL(url);
    };
  });
</script>

<svelte:head><meta name="description" content="Generate video locally with LTX 2.5. Your images, prompts and results stay on your Mac."/></svelte:head>
<div class="shell">
  <header><a class="brand" href="/" aria-label="LTX local studio home"><span class="brand-icon">L</span> LTX <span class="muted">/</span> Local studio</a><span class="local-label"><span class:online={connected && modelReady} class="dot"></span> Apple Silicon · 2.5</span></header>
  <main>
    <div class="intro"><div><p class="eyebrow">IMAGE TO VIDEO</p><h1>A still. A story.<br/><span>Set it in motion.</span></h1><p class="lede">Create with LTX 2.5, right on your Mac.</p></div><div class="privacy"><span>↗</span><p>Local by design.<br/><small>Your media stays on this machine.</small></p></div></div>
    <div class="workspace">
      <form onsubmit={generate}>
        <div class="section-heading"><h2><span>01</span> Compose your scene</h2><span class="save-note" aria-live="polite">{saved}</span></div>
        <div class="images">
          <div class="image-field"><div class="field-heading">Starting frame <span>Required</span></div><label class:has-image={firstUrl} class="upload">{#if firstUrl}<img src={firstUrl} alt="Starting frame preview"/><span class="replace">Replace image</span>{:else}<span class="upload-icon">＋</span><strong>Choose your starting image</strong><small>PNG, JPEG or WebP · up to 20MB</small>{/if}<input aria-label="Starting frame" type="file" accept="image/png,image/jpeg,image/webp" disabled={!ready} onchange={(e) => choose(e, 'first')}/></label>{#if draft.first}<p class="filename">{draft.first.name}</p>{/if}</div>
          <div class="image-field"><div class="field-heading">Ending frame <span>Optional</span></div><label class:has-image={lastUrl} class="upload end-upload">{#if lastUrl}<img src={lastUrl} alt="Ending frame preview"/><span class="replace">Replace image</span>{:else}<span class="upload-icon">↗</span><strong>Give it a destination</strong><small>Add an image to guide the ending</small>{/if}<input aria-label="Ending frame" type="file" accept="image/png,image/jpeg,image/webp" disabled={!ready} onchange={(e) => choose(e, 'last')}/></label>{#if draft.last}<button type="button" class="text-button" onclick={() => { draft.last = null; replacePreview('last'); void saveDraft(); }}>Remove ending frame</button>{/if}</div>
        </div>
        <label class="prompt-label" for="prompt">Describe the motion</label><textarea id="prompt" bind:value={draft.prompt} oninput={saveDraft} maxlength="8000" disabled={!ready} placeholder="A slow cinematic push-in. The light shifts gently as the subject turns toward the camera…"></textarea>
        <div class="settings"><label>Format<select aria-label="Format" bind:value={draft.size} onchange={saveDraft}><option value="768x512">Landscape · 768 × 512</option><option value="512x768">Portrait · 512 × 768</option><option value="512x512">Square · 512 × 512</option></select></label><label>Duration<select aria-label="Duration" bind:value={draft.frames} onchange={saveDraft}><option value={49}>2 seconds</option><option value={121}>5 seconds</option><option value={241}>10 seconds</option></select></label><label>Seed<input aria-label="Seed" aria-describedby="seed-help" type="number" min="-1" max="2147483647" step="1" required bind:value={draft.seed} onchange={saveDraft}/></label></div>
        <p id="seed-help" class="seed-help">Seed −1 chooses a new random seed each generation. A fixed seed repeats the same variation.</p>
        <button class="generate" type="submit" disabled={!ready || !modelReady || !draft.first || !draft.prompt.trim() || busy}>{busy ? 'Generation in progress…' : 'Generate video'}<span>{busy ? '◌' : '↗'}</span></button>
        <p class="under-button">You can close this page. Your job will keep running.</p>
      </form>
      <section class="result-panel" aria-label="Generated video">
        <div class="section-heading"><h2><span>02</span> Your film</h2>{#if videoUrl}<span class="result-tag">{outputCached ? 'Saved locally' : 'Available to download · not cached'}</span>{/if}</div>
        <div class="viewer" class:has-video={videoUrl}>
          {#if videoUrl}<video src={videoUrl} controls playsinline preload="metadata"><track kind="captions"/></video>{:else}<div class="empty-film"><div class="film-mark">▷</div><h3>{busy ? 'Your scene is taking shape' : 'The next frame is yours.'}</h3><p>{busy ? 'The model is working locally. Feel free to come back later.' : 'Add an image and describe what happens next.'}</p></div>{/if}
        </div>
        {#if job}<div class="job-state" role="status"><span class="dot" class:online={job.status === 'completed'}></span><strong>{job.status === 'submitting' ? 'Sending your scene' : job.status === 'queued' ? 'Queued on your Mac' : job.status === 'running' ? 'Generating on your Mac' : job.status === 'completed' ? 'Generation complete' : 'Generation failed'}</strong><small>{job.id.slice(0,8)}</small></div>{/if}
        {#if job?.request}<p class="seed-used">Seed used: <strong>{job.request.seed}</strong>{job.request.requested_seed === -1 ? ' · randomly selected' : ''}</p>{/if}
        {#if videoUrl}<a class="download" href={videoUrl} download={`ltx25-${outputId}.mp4`}>↓ Download video <span>MP4</span></a>{/if}
        <div class="server-state"><span class="dot" class:online={connected && modelReady}></span><p>{modelDetail}</p></div>
        {#if log}<details><summary>Generation log</summary><pre>{log}</pre></details>{/if}
      </section>
    </div>
    {#if error}<div class="error" role="alert">{error}<button aria-label="Dismiss error" onclick={() => error = ''}>×</button></div>{/if}
  </main>
  <footer><span>LTX 2.5 · Distilled Q8</span><span>Your workspace is restored when you return.</span></footer>
</div>

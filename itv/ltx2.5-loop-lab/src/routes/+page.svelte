<script lang="ts">
  import { onMount } from 'svelte';
  import { read, write } from '../lib/cache';
  import { LoopPlayer } from '../lib/player';
  import type { Clip, Draft, LoopSet, Playback } from '../lib/types';
  import { DEFAULT_NEGATIVE_PROMPT, isLoopSet, parseLoopSet } from '../lib/types';
  import '../style.css';

  let draft: Draft = $state({ image: null, prompt: '', negativePrompt: '', idlePrompt:'The character stands idle listening, with their mouth closed, and returns to the original pose.', idleNegativePrompt:DEFAULT_NEGATIVE_PROMPT, size: '512x512' });
  let active: LoopSet | null = $state(null);
  let clips: Clip[] = $state([]);
  let savedSets: LoopSet[] = $state([]);
  const currentDuration=$derived.by(()=>active?.clips.find(c=>c.job.id===playback.current)?.job.request.seconds??10);
  let selectedSet = $state('');
  let refreshingSets = $state(false);
  let preview = $state('');
  let anchorUrl = '';
  let canvas: HTMLCanvasElement;
  let player: LoopPlayer | undefined;
  let playback: Playback = $state({ armed: false, playing: false, current: null, slot: null, elapsed: 0, transitions: 0, buffering: false });
  let ready = $state(false);
  let submitting = $state(false);
  let modelReady = $state(false);
  let serverDetail = $state('Connecting to local model…');
  let error = $state('');
  let buildNotice = $state('');
  let saveNote = $state('Restoring workspace…');
  let sound = $state(false);
  let volume = $state(.7);
  let alive = false;
  let epoch = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let healthTimer: ReturnType<typeof setTimeout> | undefined;
  const controller = new AbortController();

  function detail(cause: unknown) { return cause instanceof Error ? cause.message : String(cause); }
  async function api(path: string, init?: RequestInit) {
    const response = await fetch(`/api${path}`, { ...init, signal: controller.signal });
    if (!response.ok) {
      const result = response.headers.get('content-type')?.includes('application/json')
        ? await response.json().catch(() => null)
        : { detail: (await response.text()).slice(0, 300) };
      throw new Error(typeof result?.detail === 'string' ? result.detail : `Local server returned ${response.status}`);
    }
    return response;
  }
  function showError(operation: string, cause: unknown) {
    if (!alive) return;
    console.error(operation, cause); error = `${operation}: ${detail(cause)}`;
  }
  async function saveDraft() {
    try { await write('draft', $state.snapshot(draft)); saveNote = 'Draft saved on this browser'; }
    catch (cause) { showError('Could not save draft', cause); saveNote = 'Draft is not saved'; }
  }
  function updatePreview() {
    if (preview) URL.revokeObjectURL(preview);
    preview = draft.image ? URL.createObjectURL(draft.image) : '';
  }
  async function chooseImage(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0]; if (!file) return;
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 20 * 1024 * 1024) {
      error = 'Choose a PNG, JPEG or WebP image under 20MB.'; return;
    }
    draft.image = file; updatePreview(); await saveDraft(); input.value = '';
  }
  async function rememberSet(batch: LoopSet) {
    savedSets = [batch, ...savedSets.filter(set => set.id !== batch.id)];
    await write('saved-sets', $state.snapshot(savedSets));
  }
  async function refreshSets() {
    refreshingSets = true;
    try {
      const result = await (await api('/v1/loop-sets')).json();
      if (!Array.isArray(result.sets)) throw new Error('Invalid saved-set list from model server.');
      const latest: LoopSet[] = result.sets.map(parseLoopSet);
      savedSets = [...latest, ...savedSets.filter(set => !latest.some(item => item.id === set.id))];
      await write('saved-sets', $state.snapshot(savedSets));
    } catch (cause) { showError('Could not refresh saved sets', cause); }
    finally { refreshingSets = false; }
  }
  async function openSavedSet() {
    const saved = savedSets.find(set => set.id === selectedSet);
    if (!saved || submitting) return;
    submitting = true; error = '';
    try {
      // Images and videos already cached on this origin can be opened offline.
      const image = await read<Blob>(`anchor:${saved.id}`) ?? await (await api(`/v1/loop-sets/${saved.id}/image`)).blob();
      const keepPlaying = playback.armed;
      await activate(saved, image, keepPlaying);
      draft = { image: new File([image], 'loop-image.png', { type: image.type || 'image/png' }),
        prompt: saved.config.prompt, negativePrompt: saved.config.negative_prompt ?? '', idlePrompt:saved.config.idle_prompt??'The character stands idle listening, with their mouth closed, and returns to the original pose.', idleNegativePrompt:saved.config.idle_negative_prompt??DEFAULT_NEGATIVE_PROMPT, size: `${saved.config.width}x${saved.config.height}` };
      updatePreview(); await saveDraft();
      if (modelReady) { clearTimeout(timer); await poll(); }
    } catch (cause) { showError('Could not open saved set', cause); }
    finally { submitting = false; }
  }
  function setLabel(batch: LoopSet) {
    const prompt = batch.config.prompt.replace(/\s+/g, ' ').trim();
    return `${prompt.slice(0, 70)}${prompt.length > 70 ? '…' : ''} · ${batch.ready_count}/${batch.total} ready · ${batch.config.width}×${batch.config.height} · ${batch.id.slice(0, 6)}`;
  }
  async function health() {
    try {
      const status = await (await api('/health')).json();
      modelReady = status.model_ready; serverDetail = status.detail;
    } catch (cause) {
      if (!alive) return;
      modelReady = false; serverDetail = 'Model server offline. Cached clips can still play.';
      console.error('Loop Lab health check failed', cause);
    }
    if (alive) healthTimer = setTimeout(health, 10000);
  }
  async function syncClips(batch: LoopSet, version: number) {
    for (const entry of batch.clips) {
      if (entry.job.status !== 'completed' || clips.some(c => c.id === entry.job.id)) continue;
      let blob = await read<Blob>(`clip:${entry.job.id}`);
      if (!blob) {
        blob = await (await api(`/v1/jobs/${entry.job.id}/video`)).blob();
        await write(`clip:${entry.job.id}`, blob);
      }
      if (!alive || version !== epoch) return;
      clips = [...clips, { id: entry.job.id, slot: entry.slot, seed: entry.job.request.seed, seconds:entry.job.request.seconds, url: URL.createObjectURL(blob) }].sort((a, b) => a.slot - b.slot);
      player?.setClips($state.snapshot(clips));
    }
  }
  async function activate(batch: LoopSet, image: Blob | undefined, autoStart: boolean) {
    batch = parseLoopSet(batch);
    selectedSet = batch.id;
    await rememberSet(batch);
    if (active?.id === batch.id) {
      active = batch; await syncClips(batch, epoch); if (autoStart) player?.start(); return;
    }
    const anchor = image ?? await read<Blob>(`anchor:${batch.id}`) ?? await (await api(`/v1/loop-sets/${batch.id}/image`)).blob();
    const version = ++epoch;
    clearTimeout(timer); player?.reset();
    for (const clip of clips) URL.revokeObjectURL(clip.url);
    clips = []; if (anchorUrl) URL.revokeObjectURL(anchorUrl);
    active = batch;
    if (!alive || version !== epoch) return;
    await write(`anchor:${batch.id}`, anchor); await write('active', $state.snapshot(batch));
    anchorUrl = URL.createObjectURL(anchor);
    player?.setAnchor(anchorUrl, batch.config.width, batch.config.height);
    if (autoStart) player?.start();
    await syncClips(batch, version);
  }
  async function poll() {
    if (!active || !alive) return;
    const id = active.id, version = epoch;
    try {
      const batch = parseLoopSet(await (await api(`/v1/loop-sets/${id}`)).json());
      if (!alive || version !== epoch) return;
      active = batch; await write('active', $state.snapshot(batch)); await rememberSet(batch);
      await syncClips(batch, version);
      saveNote = `${clips.length} of ${batch.total} clips saved on this browser`;
      if (batch.status === 'cancelled') error = 'This set was stopped by a newer build. Saved clips can still play. Resume unfinished clips to generate the rest.';
      if (batch.status === 'blocked') error = 'Generation paused after a failed clip. Completed clips can still play. Retry the failed clip to continue.';
    } catch (cause) { showError('Could not update loop set', cause); }
    if (alive && version === epoch && (active?.status === 'generating' || clips.length < (active?.ready_count ?? 0))) timer = setTimeout(poll, 2500);
  }
  async function build(event: SubmitEvent) {
    event.preventDefault(); if (!draft.image || !draft.prompt.trim() || !draft.idlePrompt.trim() || submitting) return;
    submitting = true; error = ''; buildNotice = '';
    try {
      await saveDraft();
      const form = new FormData(); form.set('image', draft.image, draft.image.name); form.set('prompt', draft.prompt); form.set('negative_prompt', draft.negativePrompt);form.set('idle_prompt',draft.idlePrompt);form.set('idle_negative_prompt',draft.idleNegativePrompt);
      const [width, height] = draft.size.split('x'); form.set('width', width); form.set('height', height);
      const batch = parseLoopSet(await (await api('/v1/loop-sets', { method: 'POST', body: form })).json());
      await navigator.storage?.persist?.();
      await activate(batch, draft.image, true); clearTimeout(timer); await poll();
      if (batch.replaced_jobs) buildNotice = `Stopped ${batch.replaced_jobs} older unfinished jobs. This set now has the model.`;
    } catch (cause) { showError('Could not build loop set', cause); }
    finally { submitting = false; }
  }
  const canRegenerateIdle=$derived.by(()=>active?.total===11&&active.clips.filter(c=>c.job.request.kind==='talking'&&c.job.status==='completed').length===7);
  async function regenerateIdle() {
    if (!active || !canRegenerateIdle || submitting || !draft.idlePrompt.trim()) return;
    const sourceId=active.id;
    submitting=true;error='';buildNotice='';
    try {
      await saveDraft();
      const form=new FormData();form.set('idle_prompt',draft.idlePrompt);form.set('idle_negative_prompt',draft.idleNegativePrompt);form.set('request_id',crypto.randomUUID());
      const batch=parseLoopSet(await (await api(`/v1/loop-sets/${sourceId}/regenerate-idle`,{method:'POST',body:form})).json());
      const image=await read<Blob>(`anchor:${sourceId}`)??await(await api(`/v1/loop-sets/${sourceId}/image`)).blob();
      await activate(batch,image,playback.armed);clearTimeout(timer);await poll();
      buildNotice='Kept all seven talking videos. Generating four fresh idle videos with your updated idle prompts. The previous set is still in Saved loop sets.';
    } catch(cause) { showError('Could not regenerate idle videos',cause); }
    finally { submitting=false; }
  }
  async function retry() {
    if (!active) return;
    try {
      error = ''; active = parseLoopSet(await (await api(`/v1/loop-sets/${active.id}/retry`, { method: 'POST' })).json());
      clearTimeout(timer); await poll();
    } catch (cause) { showError('Could not retry generation', cause); }
  }
  async function togglePlayback() {
    if (playback.armed) player?.stop();
    else {
      try { await player?.setAudio(sound, volume); player?.start(); }
      catch (cause) { showError('Could not start playback', cause); }
    }
  }
  async function toggleSound() {
    try { await player?.setAudio(!sound, volume); sound = !sound; }
    catch (cause) { showError('Could not enable audio', cause); }
  }
  async function changeVolume() {
    try { await player?.setAudio(sound, volume); }
    catch (cause) { showError('Could not change audio volume', cause); }
  }
  onMount(() => {
    alive = true;
    player = new LoopPlayer(canvas, value => playback = value, message => error = message);
    void (async () => {
      try {
        const history = await read<unknown>('saved-sets');
        savedSets = Array.isArray(history) ? history.filter(isLoopSet) : [];
        const stored = await read<Draft>('draft'); if (!alive) return;
        if (stored) draft = { ...draft, ...stored, negativePrompt:stored.negativePrompt??'', idlePrompt:stored.idlePrompt??draft.idlePrompt, idleNegativePrompt:stored.idleNegativePrompt??DEFAULT_NEGATIVE_PROMPT }; updatePreview();
        const cached = await read<unknown>('active');
        let batch = isLoopSet(cached) ? cached : undefined;
        const recovering = cached !== undefined && !batch;
        if (recovering) {
          console.error('Invalid cached Loop Lab set metadata; preserving it and recovering from the model server.');
          await write('invalid-active', cached);
        }
        // A different dev port has a different IndexedDB origin. The model
        // server is authoritative, so recover its latest set on a fresh origin.
        if (!batch && (!stored || recovering)) {
          try {
            const recent: { sets: LoopSet[] } = await (await api('/v1/loop-sets')).json();
            if (!Array.isArray(recent.sets)) throw new Error('Invalid saved-set list from model server.');
            batch = recent.sets.length ? parseLoopSet(recent.sets[0]) : undefined;
            if (batch && !stored) {
              const image = await (await api(`/v1/loop-sets/${batch.id}/image`)).blob();
              draft = { image: new File([image], 'loop-image.png', { type: 'image/png' }),
                prompt: batch.config.prompt, negativePrompt: batch.config.negative_prompt ?? '', idlePrompt:batch.config.idle_prompt??'The character stands idle listening, with their mouth closed, and returns to the original pose.', idleNegativePrompt:batch.config.idle_negative_prompt??DEFAULT_NEGATIVE_PROMPT, size: `${batch.config.width}x${batch.config.height}` };
              await write(`anchor:${batch.id}`, image); await saveDraft(); updatePreview();
            }
          } catch (cause) { showError('Could not recover saved sets from the server', cause); }
        }
        if (batch) await activate(batch, undefined, false);
        if (recovering) error = batch
          ? 'Recovered your saved clips from the server. Your draft and videos were kept.'
          : 'Saved clip metadata could not be recovered. Your draft was kept; build the set again.';
        if (!alive) return;
        ready = true; saveNote = 'Workspace restored';
        await health(); await refreshSets(); if (active) await poll();
      } catch (cause) { showError('Could not restore workspace', cause); }
    })();
    return () => {
      alive = false; epoch++; clearTimeout(timer); clearTimeout(healthTimer); controller.abort(); player?.destroy();
      if (preview) URL.revokeObjectURL(preview); if (anchorUrl) URL.revokeObjectURL(anchorUrl);
      for (const clip of clips) URL.revokeObjectURL(clip.url);
    };
  });
</script>

<svelte:head><title>Loop Lab · LTX 2.5</title><meta name="description" content="One image, talking and idle loops, infinite local video playback."/></svelte:head>
<div class="shell">
  <header><a href="/" class="brand"><span class="mark">∞</span> LOOP <span class="light">/ LAB</span></a><div class="local"><span class="dot" class:online={modelReady}></span> LTX 2.5 · On your Mac</div></header>
  <main>
    <div class="intro"><div><p class="eyebrow">AN EXPERIMENT IN CONTINUITY</p><h1>One frame.<br/><span>Endless motion.</span></h1><p class="lede">Seven talking lengths. Four idle variations. One continuous scene.</p></div><div class="recipe"><span>01 image</span><i>→</i><span>11 loops</span><i>→</i><strong>∞ possibilities</strong></div></div>
    <div class="studio">
      <aside class="composer">
        <div class="heading"><h2>Build your loop set</h2><span class="step">01</span></div>
        <div class="saved-library">
          <div class="library-heading"><label for="saved-set">Saved loop sets</label><button type="button" onclick={refreshSets} disabled={!ready || submitting || refreshingSets}>{refreshingSets ? 'Refreshing…' : 'Refresh'}</button></div>
          <select id="saved-set" bind:value={selectedSet} disabled={!ready || submitting || !savedSets.length}>
            <option value="" disabled>{savedSets.length ? 'Choose a saved scene' : 'No saved sets yet'}</option>
            {#each savedSets as batch (batch.id)}<option value={batch.id}>{setLabel(batch)}</option>{/each}
          </select>
          <button class="open-set" type="button" onclick={openSavedSet} disabled={!ready || submitting || !selectedSet}>{submitting ? 'Loading…' : 'Open saved set'}</button>
          <p>Restores the image, prompt and clips. Replaces the draft in this editor.</p>
        </div>
        <form onsubmit={build}>
          <fieldset disabled={!ready || submitting}>
            <label class="upload" class:filled={preview}>
              {#if preview}<img src={preview} alt="Shared first and last frame"/><span class="replace">Replace image</span>{:else}<span class="upload-symbol">＋</span><strong>Drop into a new scene</strong><small>Choose an image · PNG, JPEG, WebP</small>{/if}
              <input aria-label="Loop image" type="file" accept="image/png,image/jpeg,image/webp" onchange={chooseImage}/>
            </label>
            <p class="image-note">The same image anchors the start and end of every clip.</p>
            <label for="prompt">Talking prompt</label><textarea id="prompt" bind:value={draft.prompt} oninput={saveDraft} maxlength="8000" placeholder="A gentle breeze moves through the scene. The subject smiles, looks around, then returns to the original pose…"></textarea>
            <label for="negative-prompt" class="negative-label">Talking negative prompt <span>What to avoid</span></label>
            <textarea id="negative-prompt" class="negative-prompt" bind:value={draft.negativePrompt} oninput={saveDraft} maxlength="8000" placeholder="singing, exaggerated expressions"></textarea>
            <p class="negative-note">Discourage these actions. Leave empty to turn negative guidance off.</p>
            <label for="idle-prompt">Idle prompt</label><textarea id="idle-prompt" bind:value={draft.idlePrompt} oninput={saveDraft} maxlength="8000"></textarea>
            <label for="idle-negative-prompt">Idle negative prompt</label><textarea id="idle-negative-prompt" class="negative-prompt" bind:value={draft.idleNegativePrompt} oninput={saveDraft} maxlength="8000"></textarea>
            <button type="button" class="open-set" onclick={regenerateIdle} disabled={!modelReady||!canRegenerateIdle||!draft.idlePrompt.trim()}>{submitting?'Preparing…':'Regenerate idle videos'}</button>
            <p class="negative-note">Uses these idle prompts with the open set’s original image and seven completed talking clips. Creates four new 5-second idle videos. The previous set stays saved.</p>
            <div class="format-row"><label>Format<select aria-label="Format" bind:value={draft.size} onchange={saveDraft}><option value="512x512">Square · 512 × 512</option><option value="768x512">Landscape · 768 × 512</option><option value="512x768">Portrait · 512 × 768</option></select></label><div class="duration"><small>LOOP BANK</small><strong>Talking 3–9s · Idle 5s</strong></div></div>
            <button class="build" type="submit" disabled={!modelReady || !draft.image || !draft.prompt.trim() || !draft.idlePrompt.trim()}>{submitting ? 'Preparing loop set…' : 'Build & start looping'}<span>↗</span></button>
          </fieldset>
        </form>
        <p class="reuse-note">New builds stop older unfinished generations immediately. Completed clips are kept. All eleven clips generate one at a time: talking 3–9 seconds, then four 5-second idle clips.</p>
        {#if buildNotice}<p class="save-note" role="status">{buildNotice}</p>{/if}
        <p class="save-note" aria-live="polite">{saveNote}</p>
      </aside>
      <section class="playback" aria-label="Infinite video player">
        <div class="heading"><h2>Your infinite scene</h2><span class="mode-tag">RANDOM PLAY</span></div>
        <div class="stage" class:has-clips={clips.length > 0}>
          <canvas bind:this={canvas} width="512" height="512" aria-label="Infinite loop playback" data-current-clip={playback.current ?? ''} data-transitions={playback.transitions}></canvas>
          {#if !clips.length}<div class="stage-overlay"><span class="infinity">∞</span><h3>{active?.status === 'cancelled' ? 'This set was stopped.' : active ? 'Your first loop is on its way.' : 'A scene without an ending.'}</h3><p>{active?.status === 'cancelled' ? 'Resume unfinished clips to generate this scene.' : active ? 'Playback starts as soon as the first clip is ready.' : 'Add one image. Describe the motion. Let it unfold.'}</p></div>{/if}
          <div class="stage-badge"><span class="dot" class:online={playback.playing}></span>{playback.playing && playback.slot !== null ? `PLAYING LOOP ${String(playback.slot + 1).padStart(2, '0')}` : playback.armed ? 'WAITING FOR FIRST CLIP' : 'READY WHEN YOU ARE'}</div>
        </div>
        <div class="transport">
          <button class="play-button" onclick={togglePlayback} disabled={!ready}>{playback.armed ? '■ Stop' : '▶ Start'}</button>
          <div class="clock"><strong>{String(Math.min(currentDuration, Math.floor(playback.elapsed))).padStart(2, '0')}</strong><span>/ {currentDuration}s</span></div>
          <progress max={currentDuration} value={Math.min(currentDuration, playback.elapsed)} aria-label="Current loop progress"></progress>
          <button class="audio-button" aria-pressed={sound} onclick={toggleSound}>{sound ? '♪ Audio on' : '♪ Audio off'}</button>
          <input class="volume" aria-label="Volume" type="range" min="0" max="1" step=".01" bind:value={volume} oninput={changeVolume}/>
        </div>
        <div class="playback-foot"><span>{clips.length > 1 ? 'A different clip at every join' : clips.length === 1 ? 'Looping one clip while the others generate' : 'Playback begins with the first completed clip'}</span><span>{playback.transitions} joins</span></div>
      </section>
    </div>
    <section class="collection" aria-label="Loop collection">
      <div class="collection-heading"><div><p class="eyebrow">THE COLLECTION</p><h2>Talking and idle motion to take anywhere.</h2></div><span class="count">{clips.length}<span> / {active?.total??11} ready</span></span></div>
      <p>Download the source image, all videos, prompts and timing metadata in one portable file.</p>
      <a class="retry" href={active?`/api/v1/loop-sets/${active.id}/archive`:undefined} download aria-disabled={active?.status!=='completed'} onclick={event=>{if(active?.status!=='completed')event.preventDefault()}}>Download all · ZIP {active?.status==='completed'?'':'(available when complete)'}</a>
      <div class="clip-grid">
        {#each Array.from({length:active?.total??11},(_,slot)=>slot) as slot}
          {@const entry = active?.clips.find(c => c.slot === slot)}
          {@const cached = clips.find(c => c.slot === slot)}
          <article class="clip-card" class:current={playback.playing && playback.slot === slot} class:complete={!!cached}>
            <div class="card-top"><span>{entry?.job.request.kind??(slot<7?'talking':'idle')} · {entry?.job.request.seconds??(active?.total===5?10:slot<7?slot+3:5)}s</span><span>{cached ? '✓' : entry?.job.status === 'running' ? '◌' : '○'}</span></div>
            <div class="card-status">{cached ? playback.playing && playback.slot === slot ? 'Now playing' : 'Ready to play' : entry?.job.status === 'running' ? 'Generating…' : entry?.job.status === 'failed' ? 'Generation failed' : entry?.job.status === 'cancelled' ? 'Cancelled by newer build' : active ? slot === 0 ? 'Queued' : `After loop ${String(slot).padStart(2, '0')}` : 'Waiting for a scene'}</div>
            <div class="card-bottom"><small>{entry ? `Seed ${entry.job.request.seed}` : 'New seed'}</small>{#if cached}<a href={cached.url} download={`loop-${slot+1}-${cached.id}.mp4`} aria-label={`Download loop ${slot + 1}`}>↓ Save</a>{:else if entry?.job.status === 'failed'}<a href={`/api/v1/jobs/${entry.job.id}/log`} target="_blank" rel="noreferrer">View log ↗</a>{/if}</div>
          </article>
        {/each}
      </div>
      {#if active?.status === 'blocked' || active?.status === 'cancelled'}<button class="retry" onclick={retry}>{active.status === 'cancelled' ? 'Resume unfinished clips' : 'Retry failed clip'}</button>{/if}
    </section>
    {#if error}<div class="error" role="alert"><span>{error}</span><button aria-label="Dismiss error" onclick={() => error = ''}>×</button></div>{/if}
    <div class="server-note"><span class="dot" class:online={modelReady}></span>{serverDetail}</div>
  </main>
  <footer><span>LOOP LAB / LTX 2.5</span><span>Buffered joins. Fresh variations. All local.</span></footer>
</div>

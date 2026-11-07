<script lang="ts">
  import { onMount } from 'svelte';
  import { FaceRenderer } from '#lib/renderer.ts';
  import { ShapeSampler,parseMapping,sampleTrack } from '#lib/shape.ts';
  import { InferenceClient } from '#lib/inference.ts';
  import { decodeMono16k,playbackWav } from '#lib/audio.ts';
  import { generateSpeech,releaseSpeech } from '#lib/speech.ts';
  import type { AvatarMesh,Catalog,Runtime,Track } from '#lib/types.ts';

  let canvas:HTMLCanvasElement;
  let audio:HTMLAudioElement;
  let catalog=$state<Catalog|null>(null);
  let avatarId=$state('0c72c586');
  let avatarName=$state('Loading avatar');
  let mapping=$state<'linear'|'local'>('linear');
  let runtime=$state<Runtime>('webgpu');
  let sourceMode=$state<'sample'|'text'|'upload'>('sample');
  let sampleId=$state('');
  let text=$state('Mum packed a blue paper bag. We may meet Amanda near the main entrance this evening.');
  let file=$state<File|null>(null);
  let busy=$state(false);
  let avatarLoading=$state(true);
  let status=$state('Loading the visual sandbox…');
  let error=$state('');
  let progress=$state(0);
  let wireframe=$state(false);
  let closeup=$state(false);
  let contact=$state(true);
  let cavity=$state(true);
  let strength=$state(1);
  let cheekStrength=$state(1);
  let fps=$state(0);
  let track=$state<Track|null>(null);
  let playing=$state(false);
  let audioUrl=$state('');
  let preparedLabel=$state('');
  let fitHash=$state('');
  let mounted=false;
  let neutralHold=true;
  let renderer:FaceRenderer|undefined;
  let sampler:ShapeSampler|undefined;
  let client:InferenceClient;
  let avatarAbort:AbortController|undefined;
  let generationAbort:AbortController|undefined;
  const weights=new Float32Array(52);

  const currentSample=$derived(catalog?.samples.find(s=>s.id===sampleId));
  const currentAvatar=$derived(catalog?.avatars.find(a=>a.id===avatarId));
  const quickAvatars=$derived(catalog?.avatars.filter(a=>['0c72c586','01d55f04','dd80126a','05a87620'].includes(a.id))??[]);

  async function json(url:string,signal?:AbortSignal){
    const response=await fetch(url,{signal});
    if(!response.ok)throw new Error(`Could not load ${url} (HTTP ${response.status}). Run the asset preparation command in the README.`);
    return response.json();
  }
  async function loadAvatar(id=avatarId,kind=mapping){
    if(!catalog||!mounted)return;
    audio?.pause();avatarAbort?.abort();avatarAbort=new AbortController();
    const signal=avatarAbort.signal;
    avatarLoading=true;error='';avatarId=id;mapping=kind;
    renderer?.dispose();renderer=undefined;sampler=undefined;
    const summary=catalog.avatars.find(a=>a.id===id);
    if(!summary){error='That avatar is not in this local asset collection.';avatarLoading=false;return;}
    status=`Loading ${summary.name}…`;
    let next:FaceRenderer|undefined;
    try{
      const mesh=await json(summary.url,signal) as AvatarMesh;
      const map=parseMapping(await json(`/assets/avatars/${id}/${kind}.json`,signal));
      signal.throwIfAborted();next=new FaceRenderer(canvas,mesh);await next.load(signal);signal.throwIfAborted();
      if(!mounted){next.dispose();return;}
      renderer=next;sampler=new ShapeSampler(mesh,map,catalog.contact);avatarName=summary.name;fitHash=mesh.fitHash;
      status=track?'Ready. Press play to inspect the same speech on this face.':'Choose speech, then prepare the animation.';
      avatarLoading=false;
    }catch(cause){
      next?.dispose();
      if(signal.aborted)return;
      error=cause instanceof Error?cause.message:String(cause);avatarLoading=false;
      console.error('[2d-face-lab] avatar preparation failed',cause);
    }
  }
  async function prepare(){
    if(!catalog||busy||avatarLoading)return;
    generationAbort?.abort();generationAbort=new AbortController();const signal=generationAbort.signal;
    audio.pause();busy=true;progress=0;error='';playing=false;neutralHold=true;
    try{
      let pcm:Float32Array;let label='';
      if(sourceMode==='sample'){
        const selected=catalog.samples.find(s=>s.id===sampleId);if(!selected)throw new Error('Choose a speech sample.');
        status='Loading the speech sample…';const response=await fetch(selected.url,{signal});
        if(!response.ok)throw new Error(`Could not load the speech sample (HTTP ${response.status}).`);
        pcm=await decodeMono16k(await response.arrayBuffer());label=selected.text;
      }else if(sourceMode==='text'){
        pcm=await generateSpeech(text,signal,message=>{if(!signal.aborted)status=message;});label=text;
      }else{
        if(!file)throw new Error('Choose an audio file first.');
        if(file.size>20*1024*1024)throw new Error('Choose an audio file smaller than 20 MB.');
        status='Decoding your audio…';pcm=await decodeMono16k(await file.arrayBuffer());label=file.name;
      }
      signal.throwIfAborted();const sound=playbackWav(pcm);
      const result=await client.run(sound.pcm,catalog.model,runtime,signal,(message,percent)=>{if(!signal.aborted){status=message;progress=percent;}});
      signal.throwIfAborted();if(!mounted)return;
      const old=audioUrl;audioUrl=URL.createObjectURL(sound.blob);audio.src=audioUrl;audio.load();
      if(old)URL.revokeObjectURL(old);
      track=result;preparedLabel=label;neutralHold=false;progress=100;
      status='Ready. Shape follows the audio clock, including pause and seek.';
      try{await audio.play();}catch{status='Ready. Press Play in the audio controls below.';}
    }catch(cause){
      if(!signal.aborted){error=cause instanceof Error?cause.message:String(cause);status='Preparation failed.';console.error('[2d-face-lab] speech preparation failed',cause);}
    }finally{if(generationAbort?.signal===signal)busy=false;}
  }
  function stop(){
    generationAbort?.abort();client?.cancel();busy=false;audio?.pause();
    if(audio)audio.currentTime=0;
    playing=false;neutralHold=true;progress=0;status='Stopped. The face is at its own reference shape.';
  }
  onMount(()=>{
    mounted=true;client=new InferenceClient();
    runtime=new URLSearchParams(location.search).get('runtime')==='cpu'?'cpu':'webgpu';
    const boot=new AbortController();
    void(async()=>{
      try{
        const data=await json('/assets/catalog.json',boot.signal) as Catalog;
        if(!Array.isArray(data.avatars)||!data.avatars.length||!Array.isArray(data.samples))throw new Error('The local asset catalog is invalid.');
        catalog=data;sampleId=data.samples[0]?.id??'';
        if(!data.avatars.some(a=>a.id===avatarId))avatarId=data.avatars[0].id;
        await loadAvatar();
      }catch(cause){if(!boot.signal.aborted){avatarLoading=false;error=cause instanceof Error?cause.message:String(cause);console.error('[2d-face-lab] startup failed',cause);}}
    })();
    let frame=0;let count=0;let last=performance.now();
    const tick=(now:number)=>{
      if(sampler&&renderer){
        if(track&&!neutralHold)sampleTrack(track.frames,audio.currentTime,weights);
        else weights.set(sampler.map.sourceReference);
        const points=sampler.sample(weights,strength,contact,cheekStrength);
        renderer.draw(points,{wireframe,closeup,cavity});
        count++;
      }
      if(now-last>1000){fps=Math.round(count*1000/(now-last));count=0;last=now;}
      frame=requestAnimationFrame(tick);
    };
    frame=requestAnimationFrame(tick);
    return()=>{
      mounted=false;boot.abort();avatarAbort?.abort();generationAbort?.abort();client.cancel();
      cancelAnimationFrame(frame);renderer?.dispose();audio.pause();audio.removeAttribute('src');audio.load();
      if(audioUrl)URL.revokeObjectURL(audioUrl);void releaseSpeech();
    };
  });
</script>

<svelte:head>
  <title>2D Face Lab · Liforma</title>
  <meta name="description" content="A local visual sandbox for audio-driven avatar mesh animation using a single neutral texture." />
  <meta name="robots" content="noindex,nofollow" />
</svelte:head>

<div class="app-shell">
  <header class="topbar">
    <a class="brand" href="/" aria-label="Liforma face lab home"><span class="brand-mark">l</span><span><span class="wordmark">liforma</span><span class="brand-divider">/</span><strong>2D Face Lab</strong></span></a>
    <div class="experiment-tag"><span></span>Local experiment</div>
  </header>

  <main>
    <div class="intro"><div><p class="eyebrow">SPEECH → SHAPE</p><h1>A face, one texture,<br />a new way to move.</h1></div><p class="intro-note">A simple place to inspect audio-driven geometry.<br />No viseme texture blending. No production player.</p></div>
    <div class="workspace">
      <section class="viewer" aria-label="Avatar animation viewer">
        <div class="viewer-top"><div><span class="live-dot" class:active={playing}></span><strong>{avatarName}</strong><span class="viewer-caption">Neutral texture</span></div><span class="fps">{fps || '—'} fps</span></div>
        <div class="stage" class:zoomed={closeup}>
          <canvas bind:this={canvas} aria-label={`${avatarName} animated face mesh`}></canvas>
          {#if avatarLoading}<div class="stage-loading"><span class="spinner"></span><span>Loading face…</span></div>{/if}
          <div class="stage-badge"><span></span>{wireframe?'Mesh overlay':'Single texture'}</div>
          <button class="zoom-button" class:pressed={closeup} onclick={()=>closeup=!closeup} aria-pressed={closeup} title="Toggle mouth close-up"><span aria-hidden="true">⌕</span>{closeup?'Full face':'Mouth close-up'}</button>
        </div>
        <div class="transport">
          <div class="transport-title"><span>PLAYBACK</span><span>{track?`${track.duration.toFixed(1)} s · ${track.count} shape frames`:'Prepare speech to begin'}</span></div>
          <audio bind:this={audio} controls onplay={()=>{playing=true;neutralHold=false;}} onpause={()=>playing=false} onended={()=>playing=false}></audio>
          <p class="transcript">{preparedLabel||'Pause or scrub the audio to inspect the mouth at any moment.'}</p>
        </div>
      </section>

      <aside class="controls" aria-label="Animation controls">
        <section class="control-section avatar-section">
          <div class="section-heading"><h2>Choose a face</h2><span>{catalog?.avatars.length??0} avatars</span></div>
          <div class="avatar-grid">
            {#each quickAvatars as avatar}
              <button class="avatar-card" class:selected={avatar.id===avatarId} onclick={()=>loadAvatar(avatar.id)} disabled={busy} aria-pressed={avatar.id===avatarId} title={avatar.name}>
                <div class="avatar-thumb"><img src={avatar.thumbnail} alt="" /></div><span>{avatar.name.split(' ')[0]}</span>
              </button>
            {/each}
          </div>
          <label class="sr-only" for="avatar-select">All avatars</label>
          <select id="avatar-select" value={avatarId} onchange={event=>loadAvatar(event.currentTarget.value)} disabled={busy}>
            {#each catalog?.avatars??[] as avatar}<option value={avatar.id}>{avatar.name}</option>{/each}
          </select>
        </section>

        <section class="control-section">
          <div class="section-heading"><h2>Give it a voice</h2></div>
          <div class="source-tabs" aria-label="Speech source">
            <button class:chosen={sourceMode==='sample'} onclick={()=>sourceMode='sample'} disabled={busy}>Sample</button>
            <button class:chosen={sourceMode==='text'} onclick={()=>sourceMode='text'} disabled={busy}>Type text</button>
            <button class:chosen={sourceMode==='upload'} onclick={()=>sourceMode='upload'} disabled={busy}>Upload</button>
          </div>
          {#if sourceMode==='sample'}
            <label class="field-label" for="sample-select">Speech sample</label>
            <select id="sample-select" bind:value={sampleId} disabled={busy}>{#each catalog?.samples??[] as sample}<option value={sample.id}>{sample.label} · {sample.duration.toFixed(1)} s</option>{/each}</select>
            <p class="sample-copy">“{currentSample?.text??'Loading speech samples…'}”</p>
          {:else if sourceMode==='text'}
            <label class="field-label" for="speech-text">What should they say?</label>
            <textarea id="speech-text" bind:value={text} maxlength="500" rows="4" disabled={busy}></textarea>
            <div class="field-hint"><span>Speech is generated in your browser.</span><span>{text.length}/500</span></div>
          {:else}
            <label class="upload-zone" for="audio-file"><span class="upload-icon">↥</span><strong>{file?.name??'Choose an audio file'}</strong><span>WAV, MP3 or another supported format<br />Up to 30 seconds · 20 MB</span></label>
            <input id="audio-file" type="file" accept="audio/*" disabled={busy} onchange={event=>file=event.currentTarget.files?.[0]??null} />
          {/if}
          <div class="primary-actions"><button class="prepare-button" onclick={prepare} disabled={busy||avatarLoading||!catalog}><span aria-hidden="true">{busy?'◌':'▶'}</span>{busy?'Preparing…':sourceMode==='text'?'Generate & play':'Animate speech'}</button><button class="stop-button" onclick={stop} disabled={!busy&&!track} aria-label="Stop and reset" title="Stop and reset">■</button></div>
          <div class="status-line" aria-live="polite"><span class:working={busy}></span>{status}</div>
          {#if busy}<progress value={progress} max="100" aria-label="Animation preparation progress"></progress>{/if}
          {#if error}<div class="error-message" role="alert">{error}</div>{/if}
        </section>

        <section class="control-section motion-section">
          <div class="section-heading"><h2>Inspect the movement</h2></div>
          <label class="field-label" for="mapping-select">Shape mapping</label>
          <select id="mapping-select" value={mapping} onchange={event=>loadAvatar(avatarId,event.currentTarget.value as 'linear'|'local')} disabled={busy}><option value="linear">Geometry + robust linear</option><option value="local">Local nonlinear residual</option></select>
          <div class="range-label"><label for="strength">Shape strength</label><span>{Math.round(strength*100)}%</span></div>
          <input id="strength" type="range" min="0" max="1.5" step="0.01" bind:value={strength} />
          <div class="range-label"><label for="cheek-strength">Cheek transfer</label><span>{Math.round(cheekStrength*100)}%</span></div>
          <input id="cheek-strength" type="range" min="0" max="1" step="0.01" bind:value={cheekStrength} />
          <p class="quiet-note">Restores source-rig cheek motion to this face. Set to zero to compare the previous mapping.</p>
          <label class="toggle-row"><span>Show mesh</span><input type="checkbox" bind:checked={wireframe} /><span class="switch" aria-hidden="true"></span></label>
          <label class="toggle-row"><span>Lip contact correction</span><input type="checkbox" bind:checked={contact} /><span class="switch" aria-hidden="true"></span></label>
          <label class="toggle-row"><span>Dark mouth cavity</span><input type="checkbox" bind:checked={cavity} /><span class="switch" aria-hidden="true"></span></label>
          <p class="quiet-note">The cavity is a flat fill. Teeth, tongue and appearance models are not part of this visual test.</p>
        </section>
        <details class="experiment-details"><summary>Experiment details <span>↗</span></summary>
          <label class="field-label" for="runtime">Shape inference</label><select id="runtime" bind:value={runtime} disabled={busy}><option value="webgpu">WebGPU</option><option value="cpu">CPU · WASM</option></select>
          <p>Compact v4 mapping baseline · 2.25 MB</p><p>478 landmarks · 52 source controls</p>
          {#if track}<p>Inference: {track.inferenceMs.toFixed(0)} ms for {track.duration.toFixed(1)} s of audio</p>{/if}
          <p class="mono">Fit: {currentAvatar?.id??'—'}<br />{fitHash.slice(0,16)}</p>
          <p class="warning-note">These mappings have not passed the production quality gate. This viewer is for checking and improving them.</p>
        </details>
      </aside>
    </div>
    <footer><span>liforma / sandbox</span><span>Audio drives shape. The neutral texture stays the same.</span><span>SvelteKit 3</span></footer>
  </main>
</div>

<style>
  @font-face{font-family:Comfortaa;src:url(/fonts/Comfortaa-Bold.woff2) format("woff2");font-weight:700;font-display:swap}.wordmark{font-family:Comfortaa,sans-serif;font-weight:700}

  :global(*){box-sizing:border-box} :global(body){margin:0;background:#101218;color:#ececf2;font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased} :global(button),:global(input),:global(select),:global(textarea){font:inherit} :global(button){cursor:pointer} :global(button:disabled){cursor:default;opacity:.45} :global(button:focus-visible),:global(select:focus-visible),:global(textarea:focus-visible),:global(input:focus-visible),:global(summary:focus-visible){outline:2px solid #b4a0ff;outline-offset:4px} :global(select),:global(textarea){width:100%;color:#e9e9f0;background:#1d2029;border:1px solid #343743;border-radius:9px;padding:11px 12px;font-size:13px} :global(select){color-scheme:dark} :global(textarea){resize:vertical;line-height:1.6} :global(button){transition:background .15s,border-color .15s,opacity .15s} .app-shell{max-width:1520px;margin:auto}.topbar{height:78px;margin:0 50px;border-bottom:1px solid #282b34;display:flex;align-items:center;justify-content:space-between}.brand{text-decoration:none;color:#eee;display:flex;align-items:center;gap:12px;font-size:15px}.brand strong{font-weight:500;color:#bebfcd}.brand-divider{margin:0 14px;color:#4a4e5d}.brand-mark{font-family:Georgia,serif;background:#c5b8ff;color:#241a47;width:29px;height:32px;display:grid;place-items:center;border-radius:9px;font-size:26px;font-weight:bold}.experiment-tag{font-size:11px;color:#a8aabc;display:flex;align-items:center;gap:8px;border:1px solid #34343f;padding:7px 10px;border-radius:6px}.experiment-tag span{height:5px;width:5px;border-radius:50%;background:#a99ae3}main{padding:0 50px}.intro{display:flex;align-items:flex-end;justify-content:space-between;padding:36px 0 28px}.eyebrow{font-size:10px;font-weight:650;letter-spacing:2.1px;color:#b09bff;margin:0 0 12px}h1{font-size:clamp(25px,2.6vw,36px);font-weight:500;letter-spacing:-1.2px;line-height:1.16;margin:0}.intro-note{font-size:12px;line-height:1.9;color:#8d90a2;margin:0 0 3px}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:24px;align-items:start}.viewer{background:#171b24;border:1px solid #2b2f3b;border-radius:14px;overflow:hidden;position:sticky;top:20px}.viewer-top{height:55px;display:flex;align-items:center;justify-content:space-between;padding:0 23px;border-bottom:1px solid #292f3c;background:#191d27}.viewer-top>div{display:flex;align-items:center;gap:9px}.viewer-top strong{font-size:13px;font-weight:500}.viewer-caption{font-size:10px;color:#8a8ea2;border-left:1px solid #3c4050;padding-left:12px;margin-left:5px}.live-dot{width:6px;height:6px;border-radius:50%;background:#777b8a}.live-dot.active{background:#88ddb7;box-shadow:0 0 10px #88ddb744}.fps{font-size:10px;color:#8e93a6;font-variant-numeric:tabular-nums}.stage{position:relative;height:clamp(380px,48vw,635px);background:#0b0f16}.stage canvas{display:block;width:100%;height:100%}.stage-loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;gap:10px;background:#10141b;color:#9e9fb0;font-size:13px}.spinner{width:16px;height:16px;border:2px solid #393645;border-top-color:#c5b8ff;border-radius:50%;animation:spin .8s linear infinite}.stage-badge{position:absolute;left:19px;bottom:18px;display:flex;gap:6px;align-items:center;font-size:10px;color:#c7cbd6;background:#131923bf;border:1px solid #344052;padding:7px 10px;border-radius:6px;backdrop-filter:blur(8px)}.stage-badge span{width:4px;height:4px;background:#b3c4c7;border-radius:50%}.zoom-button{position:absolute;right:18px;bottom:18px;border:1px solid #3a4050;background:#171d2bd9;color:#d4d7e1;font-size:10px;padding:7px 10px;border-radius:6px;display:flex;align-items:center;gap:6px}.zoom-button.pressed{background:#373047;border-color:#8c7fa8}.zoom-button span{font-size:16px;line-height:10px}.transport{padding:18px 22px;background:#191d27;border-top:1px solid #2b3040}.transport-title{display:flex;justify-content:space-between;color:#8a8fa3;font-size:10px}.transport-title>span:first-child{letter-spacing:1.5px;font-size:9px}.transport audio{width:100%;height:38px;margin-top:12px;color-scheme:dark;display:block}.transcript{font-size:11px;line-height:1.7;color:#9d9fb1;margin:13px 0 0;min-height:19px}.controls{background:#191c24;border:1px solid #2d303c;border-radius:14px;overflow:hidden}.control-section{padding:21px 22px;border-bottom:1px solid #2c303c}.section-heading{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}.section-heading h2{font-size:13px;font-weight:550;margin:0}.section-heading>span{font-size:10px;color:#878b9e}.avatar-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px}.avatar-card{border:0;background:none;color:#9c9faf;padding:0;text-align:center;font-size:10px}.avatar-thumb{height:66px;border-radius:9px;border:1px solid #373b47;background:#252833;overflow:hidden;position:relative;margin-bottom:7px}.avatar-thumb img{height:170%;width:170%;object-fit:contain;position:absolute;left:-35%;top:0}.avatar-card.selected .avatar-thumb{border-color:#b5a1ff;box-shadow:0 0 0 1px #b5a1ff}.avatar-card.selected{color:#e7ddff}.source-tabs{display:grid;grid-template-columns:repeat(3,1fr);padding:3px;background:#12151c;border:1px solid #292e3b;border-radius:8px;gap:3px;margin-bottom:16px}.source-tabs button{border:0;color:#9297aa;background:none;border-radius:5px;padding:8px 4px;font-size:11px}.source-tabs button.chosen{background:#303440;color:#eeeef5;box-shadow:0 1px 3px #0004}.field-label{font-size:10px;display:block;color:#a6a9ba;margin:13px 0 8px}.sample-copy{font-size:12px;line-height:1.8;color:#a9aabc;margin:12px 0 17px;min-height:43px}.primary-actions{display:flex;gap:8px;margin-top:15px}.prepare-button{border:1px solid #b9a7ff;background:#bdaaff;color:#221a3c;flex:1;padding:11px 10px;border-radius:8px;font-size:12px;font-weight:650;display:flex;align-items:center;justify-content:center;gap:9px}.prepare-button:hover:not(:disabled){background:#cfbfff}.prepare-button span{font-size:10px}.stop-button{border:1px solid #3b3f4d;border-radius:8px;background:#252935;color:#b0b4c4;width:41px;font-size:11px}.status-line{font-size:10px;line-height:1.6;color:#8f94a8;margin-top:12px;display:flex;gap:7px;align-items:flex-start}.status-line>span{height:4px;width:4px;flex-shrink:0;border-radius:50%;background:#696c7b;display:inline-block;margin-top:6px}.status-line>span.working{background:#b9a7ff;animation:pulse 1s infinite}progress{width:100%;height:3px;accent-color:#b9a7ff;margin-top:12px}.error-message{padding:10px;border:1px solid #774c5b;background:#392630;color:#f1bfcd;border-radius:7px;font-size:11px;line-height:1.65;margin-top:12px}.range-label{display:flex;justify-content:space-between;font-size:11px;color:#b1b5c6;margin:20px 0 10px}.range-label span{color:#cabaff;font-size:10px}input[type=range]{width:100%;height:4px;accent-color:#c2acff;margin:0 0 17px;cursor:pointer}.toggle-row{display:flex;position:relative;align-items:center;justify-content:space-between;font-size:11px;color:#b2b5c5;padding:9px 0;cursor:pointer}.toggle-row input{position:absolute;right:0;opacity:0;width:29px;height:17px;z-index:2;cursor:pointer}.switch{width:29px;height:17px;background:#3b3e4b;border-radius:20px;position:relative}.switch::after{content:'';position:absolute;width:11px;height:11px;border-radius:50%;background:#8c91a4;left:3px;top:3px;transition:transform .15s}.toggle-row input:checked+.switch{background:#635080}.toggle-row input:checked+.switch::after{background:#d0bcff;transform:translateX(12px)}.toggle-row input:focus-visible+.switch{outline:2px solid #c2acff;outline-offset:3px}.quiet-note{font-size:10px;color:#7e8499;line-height:1.7;margin:12px 0 0}.experiment-details{padding:17px 22px;font-size:10px;color:#8a90a5}.experiment-details summary{list-style:none;cursor:pointer;display:flex;justify-content:space-between;font-size:11px;color:#a2a8bc}.experiment-details p{font-size:10px;line-height:1.7;margin-bottom:4px}.experiment-details .mono{font-family:ui-monospace,monospace;color:#6e768c}.warning-note{border-top:1px solid #323643;padding-top:10px;color:#b6a3b1}.field-hint{font-size:9px;color:#7c839a;display:flex;justify-content:space-between;margin-top:7px}.upload-zone{display:flex;align-items:center;flex-direction:column;gap:8px;border:1px dashed #4c4c66;border-radius:9px;padding:20px 8px;text-align:center;cursor:pointer;background:#1e202c}.upload-zone strong{font-size:11px;font-weight:500;max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.upload-zone>span:last-child{font-size:10px;line-height:1.7;color:#878da2}.upload-icon{font-size:22px;color:#bba7f1}#audio-file{display:block;width:100%;font-size:10px;margin-top:10px;color:#9b9eb0}footer{display:flex;justify-content:space-between;padding:27px 0 30px;color:#626a80;font-size:10px}footer>span:first-child{color:#7f8598}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@keyframes spin{to{transform:rotate(360deg)}}@keyframes pulse{50%{opacity:.3}}@media(max-width:1000px){.topbar{margin:0 25px}main{padding:0 25px}.workspace{grid-template-columns:minmax(0,1fr) 310px;gap:17px}.control-section{padding:19px}.intro-note{font-size:10px}.stage{height:520px}.avatar-thumb{height:55px}.viewer-top{padding:0 17px}.viewer-caption{display:none}}@media(max-width:750px){.topbar{height:67px;margin:0 18px}.brand{font-size:13px}.brand-divider{margin:0 9px}.experiment-tag{font-size:9px;padding:6px}main{padding:0 16px}.intro{padding:27px 0 22px}.intro-note{display:none}h1{font-size:29px}.workspace{grid-template-columns:1fr;gap:17px}.viewer{position:static}.stage{height:440px}.controls{display:grid;grid-template-columns:1fr 1fr}.control-section{min-width:0}.avatar-section{grid-column:1/-1}.avatar-grid{max-width:340px}.avatar-thumb{height:70px}.motion-section{border-left:1px solid #2c303c}.experiment-details{grid-column:1/-1}.control-section{padding:20px}.transport{padding:16px}.intro{align-items:start}footer{font-size:9px;gap:20px}footer>span:last-child{display:none}}@media(max-width:480px){.controls{display:block}.motion-section{border-left:0}.stage{height:380px}.viewer-top{height:48px}.viewer-caption{display:none}.brand-mark{width:25px;height:29px}.brand{gap:9px}.experiment-tag{display:none}.transport-title{font-size:9px}.source-tabs button{font-size:12px}.quiet-note{font-size:11px}.status-line{font-size:11px}footer>span:nth-child(2){text-align:right;max-width:210px}.intro{padding-top:24px}}@media(prefers-reduced-motion:reduce){:global(*){animation:none!important;transition:none!important}}
</style>

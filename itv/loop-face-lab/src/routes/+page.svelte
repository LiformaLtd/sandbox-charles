<script lang="ts">
 import {onMount} from 'svelte';
 import {read,write} from '../lib/cache';
 import {fingerprint} from '../lib/media-types';
 import type {Workspace,ReadyVideo,LibrarySet} from '../lib/media-types';
 import type {AvatarMesh,Catalog,Track} from '../lib/types';
 import type {Playback} from '../lib/player-types';
 import {FaceTracker} from '../lib/tracking';
 import {FacePlacement,frameAt,RIGID_ANCHORS} from '../lib/pose';
 import {fitScene} from '../lib/rigid-fit';
 import type {RigidFitStats,SceneRigidFit} from '../lib/rigid-fit';
 import {smoothSceneRotation} from '../lib/rotation-smoothing';
 import type {AxisSmoothingStats} from '../lib/rotation-smoothing';
 import {createSampler} from '../lib/retarget';
 import {ShapeSampler,sampleTrack} from '../lib/shape';
 import {FaceRenderer} from '../lib/face-renderer';
 import {NeutralNeckLayer} from '../lib/neutral-neck';
 import {LoopPlayer} from '../lib/player';
 import {InferenceClient} from '../lib/inference';
 import {generateSpeech,releaseSpeech} from '../lib/speech';
 import {VOICES,DEFAULT_VOICE,getVoice} from '../lib/voices';
 import {playbackWav} from '../lib/audio';
 import '../style.css';
 let workspace:Workspace=$state({image:null,videos:[],text:'Hello. It is lovely to meet you. What would you like to talk about today?'});
 let videos:ReadyVideo[]=$state([]),sets:LibrarySet[]=$state([]),selectedSet=$state(''),selectedSlots:number[]=$state([]);
 const selected=$derived(sets.find(set=>set.id===selectedSet));
 let busy=$state(false),speechBusy=$state(false),ready=$state(false),progress=$state('Choose a saved loop set or upload your own files.'),error=$state('');
 let preview=$state(''),audioUrl=$state(''),sound=$state(true),replace=$state(true),wire=$state(false),strength=$state(1),speechTrack:Track|undefined;
 let selectedVoice=$state<string>(DEFAULT_VOICE);
 let fixedSize=$state(true),showFit=$state(false),fitStats:RigidFitStats|undefined=$state(),fitError=$state<number|undefined>(),faceScale=$state(0);
 let yawEnabled=$state(true),yawWindow=$state(300),yawStats:AxisSmoothingStats|undefined=$state();
 let pitchEnabled=$state(true),pitchWindow=$state(300),pitchStats:AxisSmoothingStats|undefined=$state();
 let rigidSolution:SceneRigidFit|undefined;
 let neckEnabled=$state(true),neckOnly=$state(false),neckCoverage=$state(.5),neckNote=$state('');
 let neckLayer:NeutralNeckLayer|undefined;
 let runtime=$state<'webgpu'|'cpu'>('webgpu');
 let canvas:HTMLCanvasElement,audio:HTMLAudioElement;
 let playback:Playback=$state({armed:false,playing:false,current:null,slot:null,elapsed:0,transitions:0,buffering:false});
 let pose=$state({roll:0,pitch:0,yaw:0,x:0,y:0}),trackingVisible=$state(false),prepared=$state(false),speechInfo=$state('');
 let alive=false,player:LoopPlayer|undefined,renderer:FaceRenderer|undefined,placement:FacePlacement|undefined,sampler:ShapeSampler|undefined;
 let config:{model:Catalog['model'];contact:Catalog['contact']},templateMesh:AvatarMesh,templateMap:Record<string,unknown>;
 const tracker=new FaceTracker(),inference=new InferenceClient();let operation:AbortController|undefined,speechOperation:AbortController|undefined;
 let lastPose=0;const weights=new Float32Array(52);
 function fail(label:string,cause:unknown){if(!alive)return;console.error(label,cause);error=`${label}: ${cause instanceof Error?cause.message:String(cause)}`}
 async function json(path:string){const response=await fetch(path);if(!response.ok){const result=await response.json().catch(()=>({detail:`HTTP ${response.status}`}));throw new Error(result.detail||`HTTP ${response.status}`)}return response.json()}
 async function blob(path:string){const response=await fetch(path);if(!response.ok)throw new Error(`Could not load saved media (HTTP ${response.status})`);return response.blob()}
 async function persist(){await write('workspace',$state.snapshot(workspace))}
 function updatePreview(){if(preview)URL.revokeObjectURL(preview);preview=workspace.image?URL.createObjectURL(workspace.image):''}
 function clearSpeech(){speechOperation?.abort();inference.cancel();audio?.pause();if(audioUrl)URL.revokeObjectURL(audioUrl);audioUrl='';if(audio){audio.removeAttribute('src');audio.load()}speechTrack=undefined;speechInfo='';player?.redraw()}
 function clearPrepared(){operation?.abort();player?.reset();renderer?.dispose();renderer=undefined;placement=undefined;sampler=undefined;neckLayer=undefined;neckNote='';fitStats=undefined;rigidSolution=undefined;yawStats=undefined;pitchStats=undefined;fitError=undefined;faceScale=0;prepared=false;trackingVisible=false;clearSpeech()}
 function setVideos(){for(const video of videos)URL.revokeObjectURL(video.url);videos=workspace.videos.map((video,slot)=>({...video,slot,seed:0,url:URL.createObjectURL(video.file),status:'Awaiting tracking'}));updatePreview()}
 async function changeImage(event:Event){
  const input=event.currentTarget as HTMLInputElement,file=input.files?.[0];if(!file)return;
  if(!['image/png','image/jpeg','image/webp'].includes(file.type)||file.size>20*1024*1024){error='Choose a PNG, JPEG or WebP image under 20MB.';return}
  clearPrepared();workspace.image=file;updatePreview();input.value='';try{await persist()}catch(e){fail('Could not save image',e)}
 }
 async function addVideos(event:Event){
  const input=event.currentTarget as HTMLInputElement,files=Array.from(input.files??[]);if(!files.length)return;
  if(files.some(file=>file.size>150*1024*1024)||files.length+workspace.videos.length>20){error='Choose up to 20 loops, each under 150MB.';return}
  busy=true;error='';clearPrepared();
  try{for(const file of files){const id=await fingerprint(file);if(!workspace.videos.some(v=>v.id===id))workspace.videos.push({id,file,name:file.name})}setVideos();await persist();progress='Files saved. Prepare face tracking to begin.'}catch(e){fail('Could not load videos',e)}finally{busy=false;input.value=''}
 }
 async function removeVideo(id:string){clearPrepared();workspace.videos=workspace.videos.filter(v=>v.id!==id);setVideos();try{await persist()}catch(e){fail('Could not save scene',e)}}
 async function refreshLibrary(){try{const result=await json('/api/v1/loop-sets');if(!Array.isArray(result.sets))throw new Error('Invalid library response');sets=result.sets;if(!selectedSet&&sets.length){selectedSet=sets[0].id;chooseSet()}}catch(e){fail('Could not read saved loop sets',e)}}
 function chooseSet(){selectedSlots=selected?.clips.filter(c=>c.job.status==='completed').map(c=>c.slot)??[]}
 async function importSet(){
  if(!selected||!selectedSlots.length||busy)return;busy=true;error='';clearPrepared();
  try{
   const image=await blob(`/api/v1/loop-sets/${selected.id}/image`),imported=[];
   for(const clip of selected.clips.filter(c=>selectedSlots.includes(c.slot)&&c.job.status==='completed')){
    progress=`Loading saved loop ${clip.slot+1}…`;const video=await blob(`/api/v1/jobs/${clip.job.id}/video`);
    const file=new File([video],`loop-${clip.slot+1}.mp4`,{type:'video/mp4'});imported.push({id:await fingerprint(file),name:file.name,file});
   }
   workspace={...workspace,image:new File([image],'source-image.png',{type:image.type||'image/png'}),videos:imported};setVideos();await persist();
   progress='Saved loops loaded. Preparing face tracking…';
  }catch(e){fail('Could not load saved loops',e);busy=false;return}
  busy=false;await prepare();
 }
 async function updateNeck(){
  if(neckLayer&&renderer){
   renderer.setNeutralGeometry(neckLayer.setExtent(neckCoverage));
   neckNote=neckLayer.limited?'The source image has limited space below the chin. A source with more neck visible will give better coverage.':neckLayer.requiredExtent>neckCoverage?'Coverage expanded to cover the largest tracked jaw drop.':'';
  }
  player?.redraw();
  try{await write('neck-settings',{enabled:neckEnabled,coverage:neckCoverage})}catch(e){fail('Could not save neck settings',e)}
 }
 async function changePoseSettings(){
  if(placement&&renderer){
   if(rigidSolution){
    const smoothed=smoothSceneRotation(placement,rigidSolution,{yaw:yawEnabled?yawWindow:undefined,pitch:pitchEnabled?pitchWindow:undefined});
    const solution=smoothed.fit;yawStats=smoothed.stats.yaw;pitchStats=smoothed.stats.pitch;fitStats=solution.stats;
    for(const entry of solution.tracks)placement.attachFit(entry.track,entry.poses);
   }
   placement.fixedSize=fixedSize;lastPose=0;
   neckLayer=new NeutralNeckLayer(placement,templateMesh.triangles);
   for(const video of videos)if(video.track)neckLayer.prepare(video.track);
   await updateNeck();
  }
  try{await write('pose-settings',{fixedSize,yawEnabled,yawWindow,pitchEnabled,pitchWindow})}catch(e){fail('Could not save face-size settings',e)}
 }
 async function prepare(){
  if(!workspace.image||!videos.length||busy)return;busy=true;error='';clearPrepared();const signal=(operation=new AbortController()).signal;
  try{
   const image=workspace.image;progress='Fitting the source face…';
   const fit=await tracker.fitImage(image,await fingerprint(image),signal);const bitmap=await createImageBitmap(image);signal.throwIfAborted();
   placement=new FacePlacement(fit,bitmap.width,bitmap.height);sampler=createSampler(placement,templateMesh,templateMap,config.contact);
   renderer=new FaceRenderer(bitmap,placement,templateMesh.triangles);bitmap.close();
   player?.setAnchor(preview,512,512);
   for(const video of videos){
    video.status='Tracking…';video.track=await tracker.video(video.file,video.id,signal,message=>progress=`${video.name}: ${message}`);signal.throwIfAborted();
    video.status=`${video.track.frames.length} frames cached${video.track.missing?` · ${video.track.missing} without a face`:''}`;
   }
   progress='Fitting a fixed-size 3D face to the cached tracking…';
   const solution=fitScene(placement,videos.map(video=>video.track!));signal.throwIfAborted();
   rigidSolution=solution;
   const smoothed=smoothSceneRotation(placement,solution,{yaw:yawEnabled?yawWindow:undefined,pitch:pitchEnabled?pitchWindow:undefined});
   const selected=smoothed.fit;yawStats=smoothed.stats.yaw;pitchStats=smoothed.stats.pitch;fitStats=selected.stats;
   for(const entry of selected.tracks)placement.attachFit(entry.track,entry.poses);
   placement.fixedSize=fixedSize;
   neckLayer=new NeutralNeckLayer(placement,templateMesh.triangles);
   for(const video of videos)neckLayer.prepare(video.track!);
   await updateNeck();
   const first=videos[0].track!;player?.setAnchor(preview,first.width,first.height);player?.setClips($state.snapshot(videos));prepared=true;
   progress='Ready. Start the loops or enter some text and generate speech.';
  }catch(e){if(!signal.aborted)fail('Could not prepare face tracking',e)}finally{if(alive)busy=false}
 }
 function decorate(ctx:CanvasRenderingContext2D,id:string,time:number){
  const video=videos.find(v=>v.id===id),track=video?.track,frame=track?frameAt(track,time):null;
  if(!renderer||!placement||!sampler||!track||!frame){trackingVisible=false;return}
  if(speechTrack&&audio&&!audio.ended&&audio.currentTime>0)sampleTrack(speechTrack.frames,audio.currentTime,weights);else weights.set(sampler.map.sourceReference);
  sampler.sample(weights,strength,true);
  const projected=placement.project(frame,track.width,track.height,sampler.local);
  if(replace){const neutral=neckEnabled?neckLayer?.project(track,time):null;const layer=renderer.draw(projected.points,track.width,track.height,wire,true,neutral,neckOnly);canvas.dataset.neutralNeck=String(!!neutral);canvas.dataset.neckExtent=String(neckLayer?.extent??0);ctx.drawImage(layer,0,0,canvas.width,canvas.height)}
  canvas.dataset.pitchSmoothed=String(fixedSize&&pitchEnabled);canvas.dataset.fittedPitch=String(projected.pitch);canvas.dataset.fittedRoll=String(projected.roll);canvas.dataset.yawSmoothed=String(fixedSize&&yawEnabled);canvas.dataset.fittedYaw=String(projected.yaw);canvas.dataset.fixedSize=String(fixedSize);canvas.dataset.rigidScale=String(projected.scale/Math.min(track.width,track.height));
  if(showFit){
   ctx.save();ctx.scale(canvas.width/track.width,canvas.height/track.height);ctx.lineWidth=.8;
   for(const i of RIGID_ANCHORS){
    const x=frame.points[i*3]*track.width,y=frame.points[i*3+1]*track.height,px=projected.points[i*3],py=projected.points[i*3+1];
    ctx.strokeStyle='#ffb19aaa';ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(px,py);ctx.stroke();
    ctx.fillStyle='#ff987c';ctx.beginPath();ctx.arc(x,y,1.4,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#8fffe5';ctx.beginPath();ctx.arc(px,py,2.7,0,Math.PI*2);ctx.stroke();
   }ctx.restore();
  }
  if(performance.now()-lastPose>150){pose=projected;fitError=projected.fitError;faceScale=projected.scale;trackingVisible=true;lastPose=performance.now();canvas.dataset.trackingFrame=String(track.times.indexOf(frame.time));canvas.dataset.speechTime=String(audio?.currentTime??0);canvas.dataset.faceMotion=String(Math.max(...weights.map((w,i)=>Math.abs(w-sampler!.map.sourceReference[i]))))}
 }
 async function changeVoice(){
  clearSpeech();
  try{getVoice(selectedVoice);await write('speech-voice',selectedVoice)}catch(e){fail('Could not save voice choice',e)}
 }
 async function speak(){
  if(!prepared||speechBusy)return;clearSpeech();speechBusy=true;error='';const signal=(speechOperation=new AbortController()).signal;
  try{
   const voice=getVoice(selectedVoice);
   const pcm=await generateSpeech(workspace.text,signal,message=>progress=message,voice.id),playback=playbackWav(pcm);
   speechTrack=await inference.run(playback.pcm,config.model,runtime,signal,(message,percent)=>progress=`${message} ${percent}%`);
   signal.throwIfAborted();audioUrl=URL.createObjectURL(playback.blob);audio.src=audioUrl;audio.muted=!sound;audio.load();
   speechInfo=`${voice.name} · ${voice.accent} · ${speechTrack.duration.toFixed(1)}s · ${speechTrack.count} animation frames · ${Math.round(speechTrack.inferenceMs)}ms face inference`;
   await audio.play();player?.start();progress='Speaking. The video loops continue independently.';await persist();
  }catch(e){if(!signal.aborted)fail('Could not generate speech',e)}finally{if(alive)speechBusy=false}
 }
 async function togglePlayback(){
  if(playback.armed){player?.stop();audio.pause()}
  else{player?.start();if(speechTrack&&!audio.ended&&audio.currentTime>0)try{await audio.play()}catch(e){fail('Could not resume speech',e)}}
 }
 onMount(()=>{
  alive=true;player=new LoopPlayer(canvas,value=>playback=value,message=>error=message,decorate);
  void(async()=>{try{
   [config,templateMesh,templateMap]=await Promise.all([json('/assets/runtime.json'),json('/assets/template-mesh.json'),json('/assets/template-map.json')]);
   const savedVoice=await read<unknown>('speech-voice');
   if(typeof savedVoice==='string'&&VOICES.some(v=>v.id===savedVoice))selectedVoice=savedVoice;
   else if(savedVoice!==undefined){console.warn('Saved Phonon voice is unavailable',savedVoice);error='Your saved voice is unavailable. Choose another voice.'}
   const poseSettings=await read<{fixedSize:boolean;yawEnabled?:boolean;yawWindow?:number;pitchEnabled?:boolean;pitchWindow?:number}>('pose-settings');
   if(typeof poseSettings?.fixedSize==='boolean')fixedSize=poseSettings.fixedSize;
   if(typeof poseSettings?.yawEnabled==='boolean')yawEnabled=poseSettings.yawEnabled;
   if(Number.isFinite(poseSettings?.yawWindow))yawWindow=Math.max(100,Math.min(800,poseSettings!.yawWindow!));
   if(typeof poseSettings?.pitchEnabled==='boolean')pitchEnabled=poseSettings.pitchEnabled;
   if(Number.isFinite(poseSettings?.pitchWindow))pitchWindow=Math.max(100,Math.min(800,poseSettings!.pitchWindow!));
   const settings=await read<{enabled:boolean;coverage:number}>('neck-settings');
   if(settings&&typeof settings.enabled==='boolean'&&Number.isFinite(settings.coverage)){neckEnabled=settings.enabled;neckCoverage=Math.max(.2,Math.min(1,settings.coverage))}
   const saved=await read<Workspace>('workspace');if(saved)workspace=saved;setVideos();ready=true;
   if(new URLSearchParams(location.search).get('runtime')==='cpu')runtime='cpu';
   await refreshLibrary();if(workspace.image&&videos.length)await prepare();
  }catch(e){fail('Could not initialize the app',e)}})();
  return()=>{alive=false;operation?.abort();speechOperation?.abort();tracker.dispose();inference.cancel();void releaseSpeech();player?.destroy();renderer?.dispose();if(preview)URL.revokeObjectURL(preview);if(audioUrl)URL.revokeObjectURL(audioUrl);for(const v of videos)URL.revokeObjectURL(v.url)};
 });
</script>

<svelte:head><title>Loop Face Lab</title><meta name="description" content="A moving video avatar with a source-image face and speech generated in your browser."/></svelte:head>
<div class="shell">
 <header><a href="/" class="brand"><span>◉</span> LOOP FACE <b>/ LAB</b></a><span class="local">BROWSER SPEECH · LOCAL FILES</span></header>
 <main>
  <div class="intro"><p class="eyebrow">MOTION FROM VIDEO. SPEECH FROM YOU.</p><h1>A familiar face.<br/><span>A new conversation.</span></h1><p>Existing loops, a single face texture, and speech that follows your words.</p></div>
  <div class="studio">
   <aside>
    <section><h2><span>01</span> Load a scene</h2>
     <label for="library">Saved Loop Lab sets</label><div class="row"><select id="library" bind:value={selectedSet} onchange={chooseSet} disabled={!ready||busy}><option value="">Choose a saved scene</option>{#each sets as set}<option value={set.id}>{set.config.prompt.slice(0,70)} · {set.ready_count}/5</option>{/each}</select><button class="subtle" onclick={refreshLibrary} disabled={busy}>Refresh</button></div>
     {#if selected}<div class="choices">{#each selected.clips as clip}<label><input type="checkbox" value={clip.slot} bind:group={selectedSlots} disabled={busy||clip.job.status!=='completed'}/>Loop {clip.slot+1}</label>{/each}</div>{/if}
     <button class="secondary full" onclick={importSet} disabled={!ready||busy||speechBusy||!selectedSlots.length}>Load selected loops & source image</button>
     <p class="hint">Loads existing files. No video generation.</p>
     <div class="divider">OR UPLOAD FROM DISK</div>
     <label class="image-input">{#if preview}<img src={preview} alt="Original source face"/>{:else}<span>＋ Original source image</span>{/if}<input aria-label="Source image" type="file" accept="image/png,image/jpeg,image/webp" onchange={changeImage} disabled={!ready||busy||speechBusy}/></label>
     <label class="file-button">＋ Add one or more video loops<input aria-label="Video loops" type="file" multiple accept="video/mp4,video/webm,video/quicktime" onchange={addVideos} disabled={!ready||busy||speechBusy}/></label>
     <p class="hint">Use loops made from this image. Up to 20 clips, each at most 30 seconds. Playback uses up to 10 seconds per clip.</p>
     {#each videos as video}<div class="video-row"><div><strong>{video.name}</strong><small>{video.status}</small></div><button aria-label={`Remove ${video.name}`} onclick={()=>removeVideo(video.id)} disabled={busy||speechBusy}>×</button></div>{/each}
     <button class="secondary full" onclick={prepare} disabled={!ready||busy||speechBusy||!workspace.image||!videos.length}>{prepared?'Reload cached tracking':'Prepare face tracking'}</button>
    </section>
    <section><h2><span>02</span> Give it something to say</h2>
     <label for="speech">Speech text</label><textarea id="speech" bind:value={workspace.text} maxlength="500" oninput={()=>persist().catch(e=>fail('Could not save text',e))} disabled={speechBusy} placeholder="What would you like your avatar to say?"></textarea>
     <label for="voice">Phonon voice</label><select id="voice" bind:value={selectedVoice} onchange={changeVoice} disabled={!ready||speechBusy}>
      <optgroup label="Female voices">{#each VOICES.filter(v=>v.gender==='female') as voice}<option value={voice.id}>{voice.name} · {voice.accent}</option>{/each}</optgroup>
      <optgroup label="Male voices">{#each VOICES.filter(v=>v.gender==='male') as voice}<option value={voice.id}>{voice.name} · {voice.accent}</option>{/each}</optgroup>
     </select>
     <p class="hint">Your last voice choice is remembered on this browser.</p>
     <label for="runtime">Face animation runtime</label><select id="runtime" bind:value={runtime} disabled={speechBusy}><option value="webgpu">WebGPU · Apple GPU</option><option value="cpu">CPU · WASM</option></select>
     <button class="primary full" onclick={speak} disabled={!prepared||busy||speechBusy||!workspace.text.trim()}>{speechBusy?'Preparing speech…':'Generate & speak'}<span>↗</span></button>
     {#if speechBusy||busy}<button class="subtle full" onclick={()=>{operation?.abort();speechOperation?.abort();inference.cancel();tracker.dispose()}}>Cancel preparation</button>{/if}
     <p class="hint">Phonon speech and compact Audio2Face run in this browser. English · up to 500 characters / 30 seconds.</p>
    </section>
   </aside>
   <section class="viewer">
    <div class="viewer-heading"><h2>Your speaking scene</h2><span class="tag">{playback.playing?'LIVE PLAYBACK':'READY WHEN YOU ARE'}</span></div>
    <div class="stage"><canvas bind:this={canvas} width="512" height="512" aria-label="Speaking avatar" data-current-clip={playback.current??''} data-transitions={playback.transitions}></canvas>{#if !preview}<div class="empty"><span>◉</span><h3>Let the scene speak.</h3><p>Load loops and their source image to begin.</p></div>{/if}</div>
    <div class="transport"><button class="primary" onclick={togglePlayback} disabled={!prepared||busy}>{playback.armed?'■ Stop':'▶ Start loops'}</button><span>Loop {playback.slot===null?'—':playback.slot+1} · {playback.elapsed.toFixed(1)}s</span><span>{playback.transitions} joins</span></div>
    <div class="toggles"><label><input type="checkbox" bind:checked={replace} onchange={()=>player?.redraw()}/> Replace face</label><label><input type="checkbox" bind:checked={wire} onchange={()=>player?.redraw()}/> Show mesh</label><label><input type="checkbox" bind:checked={sound} onchange={()=>audio.muted=!sound}/> Speech audio</label></div>
    <div class="fit-controls">
     <div class="toggles"><label><input type="checkbox" bind:checked={fixedSize} onchange={changePoseSettings} disabled={busy}/> Fixed 3D face size</label><label><input type="checkbox" bind:checked={showFit} onchange={()=>player?.redraw()}/> Show fitted points</label></div>
     <div class="toggles"><label><input type="checkbox" bind:checked={yawEnabled} onchange={changePoseSettings} disabled={busy||!fixedSize}/> Smooth yaw</label></div>
     <label class="strength">Yaw smoothing <input aria-label="Yaw smoothing" type="range" min="100" max="800" step="50" bind:value={yawWindow} oninput={changePoseSettings} disabled={busy||!fixedSize||!yawEnabled}/><span>{yawWindow}ms</span></label>
     {#if fixedSize&&yawEnabled&&yawStats}<p class="yaw-summary">Yaw roughness {Math.abs(yawStats.reduction).toFixed(0)}% {yawStats.reduction>=0?'lower':'higher'} · loop endpoints aligned</p>{/if}
     <div class="toggles"><label><input type="checkbox" bind:checked={pitchEnabled} onchange={changePoseSettings} disabled={busy||!fixedSize}/> Smooth pitch</label></div>
     <label class="strength">Pitch smoothing <input aria-label="Pitch smoothing" type="range" min="100" max="800" step="50" bind:value={pitchWindow} oninput={changePoseSettings} disabled={busy||!fixedSize||!pitchEnabled}/><span>{pitchWindow}ms</span></label>
     {#if fixedSize&&pitchEnabled&&pitchStats}<p class="yaw-summary">Pitch roughness {Math.abs(pitchStats.reduction).toFixed(0)}% {pitchStats.reduction>=0?'lower':'higher'} · loop endpoints aligned</p>{/if}
     <p class="hint">Yaw and pitch smoothing reduce fit noise without delaying playback. Roll is unsmoothed; position and fixed scale stay as fitted.</p>
     <p class="hint">One face shape and camera scale across the loaded loops. Eye corners, nose and temple points determine head rotation and position.</p>
     {#if fitStats}<p class="fit-summary">{fixedSize?'Size locked':'Per-frame size comparison'} · {fitStats.frames} fitted frames · detected size variation {fitStats.rawScaleDrift.toFixed(2)}%{#if fixedSize} · median fit error {fitStats.medianError.toFixed(2)}px{/if}</p>{/if}
     {#if showFit}<p class="hint">Coral dots: detected points. Mint rings: projected fixed 3D points. Lines show the remaining mismatch.</p>{/if}
    </div>
    <div class="neck-controls">
     <div class="toggles"><label><input type="checkbox" bind:checked={neckEnabled} onchange={updateNeck}/> Neutral face & neck</label><label><input type="checkbox" bind:checked={neckOnly} onchange={()=>player?.redraw()} disabled={!neckEnabled}/> Neutral layer only</label></div>
     <label class="strength">Neck coverage <input aria-label="Neck coverage" type="range" min=".2" max="1" step=".05" bind:value={neckCoverage} oninput={updateNeck} disabled={!neckEnabled}/><span>{Math.round(neckCoverage*100)}%</span></label>
     <p class="hint">The neutral layer covers the original jaw. Motion eases down the neck before the outer edge fades. Mesh view: amber is the neutral layer; green is speech.</p>
     {#if neckNote}<p class="warning">{neckNote}</p>{/if}
    </div>
    <label class="strength">Speech motion strength <input aria-label="Speech motion strength" type="range" min="0" max="1.5" step=".05" bind:value={strength} oninput={()=>player?.redraw()}/><span>{Math.round(strength*100)}%</span></label>
    <audio bind:this={audio} onseeked={()=>player?.redraw()} controls aria-label="Generated speech"></audio><p class="hint">{speechInfo||'Generated speech appears here. Original video audio stays muted.'}</p>
    <div class="diagnostics"><div><small>HEAD POSITION</small><strong>{pose.x.toFixed(0)}, {pose.y.toFixed(0)} px</strong></div><div><small>ROLL</small><strong>{pose.roll.toFixed(1)}°</strong></div><div><small>PITCH</small><strong>{pose.pitch.toFixed(1)}°</strong></div><div><small>YAW</small><strong>{pose.yaw.toFixed(1)}°</strong></div></div>
    <p class="fit-current">Face scale {faceScale?faceScale.toFixed(2):'—'} px / unit{#if fitError!==undefined} · current fit error {fitError.toFixed(2)} px{/if}</p>
    <p class="status" aria-live="polite">{progress}</p>
    {#if prepared&&playback.playing&&!trackingVisible}<p class="warning">No tracked face in this frame. Showing the original video.</p>{/if}
    <details><summary>About this experiment</summary><p>A neutral source-image face and extended neck are painted below the speaking mesh. The neck transition reduces head motion toward a smoothed image-plane anchor; it is not a separate torso tracker. The opaque coverage is sized from the cached jaw positions. Fixed-size fitting uses one 3D reference and an orthographic camera scale calibrated across all loaded frames. Rotation and position are fitted to the eyes, nose and temple points; speech keeps those anchor vertices fixed. Offline symmetric filters smooth yaw and pitch independently. The first and last filtered angles of each loop ease to shared values. Roll is never smoothed; position and scale are unchanged. Raw fitting remains available for comparison. Speech replaces video expressions using the existing calibrated ARKit-to-MediaPipe motion, adapted to this source fit. Tracking is cached by video content; it does not run again during playback.</p><p>This is a single-texture experiment. It cannot reconstruct unseen face surfaces, teeth or tongue, or reliably handle hands crossing the face. Strong head turns and lighting changes may reveal seams. Use the original-video and mesh views to compare.</p></details>
   </section>
  </div>
  {#if error}<div class="error" role="alert"><span>{error}</span><button aria-label="Dismiss error" onclick={()=>error=''}>×</button></div>{/if}
 </main><footer>LOOP FACE LAB <span>Video motion + source face + browser speech</span></footer>
</div>

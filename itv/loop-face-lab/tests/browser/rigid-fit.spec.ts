import {test,expect} from '@playwright/test';

test('fixed 3D size is constant during real playback, comparable with raw sizing, and persists without rerunning detection',async({page},info)=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/?runtime=cpu');await expect(page.getByLabel('Saved Loop Lab sets')).toBeEnabled();
 const library=await(await page.request.get('/api/v1/loop-sets')).json();const talking=library.sets.find((s:any)=>s.config.prompt==='The character talks and moves their hands.');
 await page.getByLabel('Saved Loop Lab sets').selectOption(talking.id);
 for(const i of[3,4,5])await page.getByRole('checkbox',{name:`Loop ${i}`,exact:true}).uncheck();
 await page.getByRole('button',{name:'Load selected loops & source image'}).click();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:90000});
 await expect(page.getByLabel('Fixed 3D face size',{exact:true})).toBeChecked();
 await expect(page.locator('.fit-summary')).toContainText('482 fitted frames');
 await page.getByRole('button',{name:'▶ Start loops'}).click();const canvas=page.getByLabel('Speaking avatar');await expect(canvas).toHaveAttribute('data-fixed-size','true');
 const sample=()=>page.evaluate(async()=>{
  const canvas=document.querySelector('canvas')!;const values:number[]=[];const start=performance.now();
  while(performance.now()-start<1500){const value=Number(canvas.dataset.rigidScale);if(value>0)values.push(value);await new Promise(resolve=>requestAnimationFrame(resolve))}
  return {count:values.length,range:Math.max(...values)-Math.min(...values)};
 });
 const fixed=await sample();expect(fixed.count).toBeGreaterThan(10);expect(fixed.range).toBe(0);
 await page.getByLabel('Fixed 3D face size',{exact:true}).uncheck();await expect(canvas).toHaveAttribute('data-fixed-size','false');
 const raw=await sample();expect(raw.range).toBeGreaterThan(.00001);
 await page.getByLabel('Fixed 3D face size',{exact:true}).check();await expect(canvas).toHaveAttribute('data-fixed-size','true');
 await page.getByLabel('Show fitted points',{exact:true}).check();
 await page.screenshot({path:info.outputPath('fixed-size-face.png'),fullPage:true});
 const result=await page.evaluate(async()=>{
  const [{FacePlacement,RIGID_ANCHORS,projectPoint},{fitScene},{fingerprint},{read}]=await Promise.all([
   import(/* @vite-ignore */ '/src/lib/pose.ts'),import(/* @vite-ignore */ '/src/lib/rigid-fit.ts'),import(/* @vite-ignore */ '/src/lib/media-types.ts'),import(/* @vite-ignore */ '/src/lib/cache.ts')]);
  const workspace=await read('workspace'),source=await read(`fit:1:${await fingerprint(workspace.image)}`),bitmap=await createImageBitmap(workspace.image),tracks=[];
  for(const video of workspace.videos)tracks.push(await read(`track:1:${video.id}`));
  const placement=new FacePlacement(source,bitmap.width,bitmap.height);bitmap.close();const fit=fitScene(placement,tracks),distances:number[]=[];
  for(const entry of fit.tracks)for(const pose of entry.poses)if(pose){
   const a=RIGID_ANCHORS[0],b=RIGID_ANCHORS[3],pa=projectPoint(placement.canonical.subarray(a*3,a*3+3),pose),pb=projectPoint(placement.canonical.subarray(b*3,b*3+3),pose);
   distances.push(Math.hypot(...pa.map((v:number,i:number)=>v-pb[i]))/pose.scale);
  }
  return {...fit.stats,anchorDistanceRange:Math.max(...distances)-Math.min(...distances),frames:fit.stats.frames};
 });
 expect(result.anchorDistanceRange).toBeLessThan(.000001);expect(result.frames).toBe(482);expect(result.medianError).toBeLessThan(8);
 await (await import('node:fs/promises')).writeFile(info.outputPath('rigid-fit.json'),JSON.stringify({...result,fixedPlayback:fixed,rawPlayback:raw,pageErrors:errors},null,2));
 await page.getByRole('button',{name:'■ Stop'}).click();
 await page.getByLabel('Fixed 3D face size',{exact:true}).uncheck();
 let detections=0;page.on('request',r=>{if(r.url().includes('face_landmarker.task'))detections++});
 await page.reload();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:20000});
 await expect(page.getByLabel('Fixed 3D face size',{exact:true})).not.toBeChecked();expect(detections).toBe(0);
 expect(errors).toEqual([]);
});

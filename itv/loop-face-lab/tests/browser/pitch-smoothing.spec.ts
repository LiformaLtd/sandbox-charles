import {test,expect} from '@playwright/test';

test('yaw and pitch smooth independently while roll stays exact, and both controls persist',async({page},info)=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/?runtime=cpu');await expect(page.getByLabel('Saved Loop Lab sets')).toBeEnabled();
 const library=await(await page.request.get('/api/v1/loop-sets')).json(),talking=library.sets.find((s:any)=>s.config.prompt==='The character talks and moves their hands.');
 await page.getByLabel('Saved Loop Lab sets').selectOption(talking.id);for(const i of[3,4,5])await page.getByRole('checkbox',{name:`Loop ${i}`,exact:true}).uncheck();
 await page.getByRole('button',{name:'Load selected loops & source image'}).click();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:90000});
 await expect(page.getByLabel('Smooth yaw',{exact:true})).toBeChecked();await expect(page.getByLabel('Smooth pitch',{exact:true})).toBeChecked();
 await page.getByRole('button',{name:'▶ Start loops'}).click();const canvas=page.getByLabel('Speaking avatar');await expect(canvas).toHaveAttribute('data-pitch-smoothed','true');
 await page.waitForTimeout(1200);await page.getByRole('button',{name:'■ Stop'}).click();
 const before=await canvas.evaluate(c=>({...c.dataset}));await page.getByLabel('Smooth pitch',{exact:true}).uncheck();await expect(canvas).toHaveAttribute('data-pitch-smoothed','false');
 const after=await canvas.evaluate(c=>({...c.dataset}));expect(after.fittedYaw).toBe(before.fittedYaw);expect(after.fittedRoll).toBe(before.fittedRoll);expect(after.rigidScale).toBe(before.rigidScale);
 await page.getByLabel('Smooth pitch',{exact:true}).check();await expect(canvas).toHaveAttribute('data-pitch-smoothed','true');
 const slider=page.getByLabel('Pitch smoothing',{exact:true});await slider.press('Home');for(let n=0;n<6;n++)await slider.press('ArrowRight');await expect(slider).toHaveValue('400');
 await expect(page.getByLabel('Yaw smoothing',{exact:true})).toHaveValue('300');
 const result=await page.evaluate(async()=>{
  const [{FacePlacement},{fitScene},{smoothSceneRotation},{fingerprint},{read}]=await Promise.all([
   import(/* @vite-ignore */ '/src/lib/pose.ts'),import(/* @vite-ignore */ '/src/lib/rigid-fit.ts'),import(/* @vite-ignore */ '/src/lib/rotation-smoothing.ts'),import(/* @vite-ignore */ '/src/lib/media-types.ts'),import(/* @vite-ignore */ '/src/lib/cache.ts')]);
  const workspace=await read('workspace'),source=await read(`fit:1:${await fingerprint(workspace.image)}`),bitmap=await createImageBitmap(workspace.image),tracks=[];
  for(const video of workspace.videos)tracks.push(await read(`track:1:${video.id}`));
  const placement=new FacePlacement(source,bitmap.width,bitmap.height);bitmap.close();const original=fitScene(placement,tracks),filtered=smoothSceneRotation(placement,original,{yaw:300,pitch:400});let rollDifference=0,otherDifference=0,endDifference=0;
  filtered.fit.tracks.forEach((entry:any,k:number)=>{
   entry.poses.forEach((p:any,n:number)=>{if(!p)return;const old=original.tracks[k].poses[n];rollDifference=Math.max(rollDifference,Math.abs(p.roll-old.roll));for(const field of['x','y','scale'])otherDifference=Math.max(otherDifference,Math.abs(p[field]-old[field]))});
   for(const axis of['yaw','pitch'])endDifference=Math.max(endDifference,Math.abs(entry.poses[0][axis]-filtered.stats[axis].endpointAngle),Math.abs(entry.poses.at(-1)[axis]-filtered.stats[axis].endpointAngle));
  });return {stats:filtered.stats,rollDifference,otherDifference,endDifference,medianFitError:filtered.fit.stats.medianError,frames:filtered.fit.stats.frames};
 });
 expect(result.stats.yaw.reduction).toBeGreaterThan(40);expect(result.stats.pitch.reduction).toBeGreaterThan(40);expect(result.rollDifference).toBe(0);expect(result.otherDifference).toBe(0);expect(result.endDifference).toBe(0);
 await page.screenshot({path:info.outputPath('rotation-smoothing.png'),fullPage:true});
 await (await import('node:fs/promises')).writeFile(info.outputPath('rotation-smoothing.json'),JSON.stringify({...result,pageErrors:errors},null,2));
 await page.reload();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:20000});await expect(slider).toHaveValue('400');await expect(page.getByLabel('Yaw smoothing',{exact:true})).toHaveValue('300');await expect(page.getByLabel('Smooth pitch',{exact:true})).toBeChecked();expect(errors).toEqual([]);
});

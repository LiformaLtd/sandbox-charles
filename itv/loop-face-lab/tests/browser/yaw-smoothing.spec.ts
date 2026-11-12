import {test,expect} from '@playwright/test';

test('yaw smoothing reduces real fit roughness, preserves other pose components and survives refresh',async({page},info)=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/?runtime=cpu');await expect(page.getByLabel('Saved Loop Lab sets')).toBeEnabled();
 const library=await(await page.request.get('/api/v1/loop-sets')).json(),talking=library.sets.find((s:any)=>s.config.prompt==='The character talks and moves their hands.');
 await page.getByLabel('Saved Loop Lab sets').selectOption(talking.id);for(const i of[3,4,5])await page.getByRole('checkbox',{name:`Loop ${i}`,exact:true}).uncheck();
 await page.getByRole('button',{name:'Load selected loops & source image'}).click();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:90000});
 await expect(page.getByLabel('Smooth yaw',{exact:true})).toBeChecked();await expect(page.getByLabel('Yaw smoothing',{exact:true})).toHaveValue('300');
 await page.getByRole('button',{name:'▶ Start loops'}).click();const canvas=page.getByLabel('Speaking avatar');await expect(canvas).toHaveAttribute('data-yaw-smoothed','true');
 await page.getByRole('button',{name:'■ Stop'}).click();await page.getByLabel('Smooth yaw',{exact:true}).uncheck();await expect(canvas).toHaveAttribute('data-yaw-smoothed','false');
 await page.getByLabel('Smooth yaw',{exact:true}).check();await expect(canvas).toHaveAttribute('data-yaw-smoothed','true');
 const result=await page.evaluate(async()=>{
  const [{FacePlacement},{fitScene},{smoothSceneYaw},{fingerprint},{read}]=await Promise.all([
   import(/* @vite-ignore */ '/src/lib/pose.ts'),import(/* @vite-ignore */ '/src/lib/rigid-fit.ts'),import(/* @vite-ignore */ '/src/lib/rotation-smoothing.ts'),import(/* @vite-ignore */ '/src/lib/media-types.ts'),import(/* @vite-ignore */ '/src/lib/cache.ts')]);
  const workspace=await read('workspace'),source=await read(`fit:1:${await fingerprint(workspace.image)}`),bitmap=await createImageBitmap(workspace.image),tracks=[];
  for(const video of workspace.videos)tracks.push(await read(`track:1:${video.id}`));
  const placement=new FacePlacement(source,bitmap.width,bitmap.height);bitmap.close();const original=fitScene(placement,tracks),filtered=smoothSceneYaw(placement,original,300);let otherPoseDifference=0,endpointDifference=0;
  filtered.fit.tracks.forEach((entry:any,k:number)=>{
   entry.poses.forEach((pose:any,n:number)=>{if(!pose)return;for(const field of['roll','pitch','x','y','scale','centerX','centerY'])otherPoseDifference=Math.max(otherPoseDifference,Math.abs(pose[field]-original.tracks[k].poses[n][field]))});
   endpointDifference=Math.max(endpointDifference,Math.abs(entry.poses[0].yaw-filtered.stats.endpointYaw),Math.abs(entry.poses.at(-1).yaw-filtered.stats.endpointYaw));
  });return {...filtered.stats,frames:filtered.fit.stats.frames,medianFitError:filtered.fit.stats.medianError,originalMedianFitError:original.stats.medianError,otherPoseDifference,endpointDifference};
 });
 expect(result.after).toBeLessThan(result.before*.6);expect(result.otherPoseDifference).toBe(0);expect(result.endpointDifference).toBe(0);
 const slider=page.getByLabel('Yaw smoothing',{exact:true});await slider.press('Home');for(let i=0;i<8;i++)await slider.press('ArrowRight');await expect(slider).toHaveValue('500');
 await page.screenshot({path:info.outputPath('yaw-smoothing.png'),fullPage:true});
 let trackingRequests=0;page.on('request',r=>{if(r.url().includes('face_landmarker.task'))trackingRequests++});await page.reload();
 await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:20000});await expect(slider).toHaveValue('500');await expect(page.getByLabel('Smooth yaw',{exact:true})).toBeChecked();
 expect(trackingRequests).toBe(0);expect(errors).toEqual([]);
 await (await import('node:fs/promises')).writeFile(info.outputPath('yaw-smoothing.json'),JSON.stringify({...result,pageErrors:errors},null,2));
});

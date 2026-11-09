import {test,expect} from '@playwright/test';

test('neutral neck covers the exposed original chin and supports paused comparison, coverage and cached settings',async({page},info)=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/?runtime=cpu');await expect(page.getByLabel('Saved Loop Lab sets')).toBeEnabled();
 const library=await(await page.request.get('/api/v1/loop-sets')).json();const talking=library.sets.find((s:any)=>s.config.prompt==='The character talks and moves their hands.');
 await page.getByLabel('Saved Loop Lab sets').selectOption(talking.id);
 for(const i of [2,3,4,5])await page.getByRole('checkbox',{name:`Loop ${i}`,exact:true}).uncheck();
 await page.getByRole('button',{name:'Load selected loops & source image'}).click();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:90000});
 await page.getByRole('button',{name:'▶ Start loops'}).click();const canvas=page.getByLabel('Speaking avatar');
 await expect(canvas).toHaveAttribute('data-neutral-neck','true');await page.getByRole('button',{name:'■ Stop'}).click();
 await page.getByLabel('Neutral face & neck',{exact:true}).uncheck();await expect(canvas).toHaveAttribute('data-neutral-neck','false');
 await page.getByLabel('Neutral face & neck',{exact:true}).check();await expect(canvas).toHaveAttribute('data-neutral-neck','true');
 const coverage=page.getByLabel('Neck coverage',{exact:true});await coverage.press('Home');for(let i=0;i<9;i++)await coverage.press('ArrowRight');
 await expect.poll(async()=>Number(await canvas.getAttribute('data-neck-extent'))).toBeGreaterThanOrEqual(.65);
 await page.getByLabel('Neutral layer only',{exact:true}).check();await page.getByLabel('Show mesh',{exact:true}).check();
 await page.screenshot({path:info.outputPath('neutral-neck-mesh.png'),fullPage:true});
 const result=await page.evaluate(async()=>{
  // Test the actual renderer at the worst chin-drop frame from the cached real clip.
  const [{FacePlacement},{NeutralNeckLayer},{FaceRenderer},{fingerprint},{read}]=await Promise.all([
   import(/* @vite-ignore */ '/src/lib/pose.ts'),import(/* @vite-ignore */ '/src/lib/neutral-neck.ts'),import(/* @vite-ignore */ '/src/lib/face-renderer.ts'),import(/* @vite-ignore */ '/src/lib/media-types.ts'),import(/* @vite-ignore */ '/src/lib/cache.ts')]);
  const workspace=await read('workspace'),fit=await read(`fit:1:${await fingerprint(workspace.image)}`),track=await read(`track:1:${workspace.videos[0].id}`),bitmap=await createImageBitmap(workspace.image),template=await(await fetch('/assets/template-mesh.json')).json();
  const placement=new FacePlacement(fit,bitmap.width,bitmap.height),neck=new NeutralNeckLayer(placement,template.triangles);neck.prepare(track);neck.setExtent(.5);
  let index=0,maxDrop=-Infinity;
  for(let n=0;n<track.frames.length;n++){const frame=track.frames[n];if(!frame)continue;const p=placement.project(frame,track.width,track.height);const drop=frame.points[152*3+1]*track.height-p.points[152*3+1];if(drop>maxDrop){index=n;maxDrop=drop}}
  const frame=track.frames[index],projected=placement.project(frame,track.width,track.height),neutral=neck.project(track,frame.time),renderer=new FaceRenderer(bitmap,placement,template.triangles);renderer.setNeutralGeometry(neck.geometry);bitmap.close();
  const copy=(source:CanvasImageSource)=>{const c=document.createElement('canvas');c.width=track.width;c.height=track.height;c.getContext('2d')!.drawImage(source,0,0);return c};
  const before=copy(renderer.draw(projected.points,track.width,track.height,false,true)),after=copy(renderer.draw(projected.points,track.width,track.height,false,true,neutral));
  const x=Math.round(frame.points[152*3]*track.width),y=Math.round(frame.points[152*3+1]*track.height);
  const oldAlpha=before.getContext('2d')!.getImageData(x,y,1,1).data[3],newAlpha=after.getContext('2d')!.getImageData(x,y,1,1).data[3];
  renderer.dispose();return {frame:index,time:frame.time,maxDrop,oldAlpha,newAlpha,extent:neck.extent};
 });
 expect(result.maxDrop).toBeGreaterThan(8);expect(result.oldAlpha).toBeLessThan(100);expect(result.newAlpha).toBe(255);
 await (await import('node:fs/promises')).writeFile(info.outputPath('chin-coverage.json'),JSON.stringify(result,null,2));
 await page.reload();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:20000});
 await expect(page.getByLabel('Neutral face & neck',{exact:true})).toBeChecked();await expect(page.getByLabel('Neck coverage',{exact:true})).toHaveValue('0.65');expect(errors).toEqual([]);
});

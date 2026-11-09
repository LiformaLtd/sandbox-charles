import {test,expect} from '@playwright/test';
// Opt-in integration: existing, generated character loops are read only.
test('imports saved clips, tracks every decoded frame, speaks in the browser, and restores cached tracking',async({page},info)=>{
 const errors:string[]=[],posts:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 page.on('console',message=>{if(message.type()==='error')console.log('BROWSER:',message.text().slice(0,800))});
 page.on('request',request=>{if(request.method()==='POST')posts.push(request.url())});
 const library=await(await page.request.get('/api/v1/loop-sets')).json(),talking=library.sets.find((s:any)=>s.config.prompt==='The character talks and moves their hands.');
 await page.goto('/?runtime=cpu');await expect(page.getByLabel('Saved Loop Lab sets')).toBeEnabled();await page.getByLabel('Saved Loop Lab sets').selectOption(talking.id);
 await expect(page.getByRole('checkbox',{name:'Loop 1',exact:true})).toBeVisible();
 for(const slot of [3,4,5])await page.getByRole('checkbox',{name:`Loop ${slot}`,exact:true}).uncheck();
 await page.getByRole('button',{name:'Load selected loops & source image'}).click();
 await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:150000});
 await expect(page.getByRole('alert')).toHaveCount(0);
 const videos=await page.locator('.video-row').allTextContents();expect(videos).toHaveLength(2);for(const text of videos)expect(text).toContain('241 frames cached');
 await page.getByRole('button',{name:'▶ Start loops'}).click();
 const canvas=page.getByLabel('Speaking avatar');await expect(canvas).not.toHaveAttribute('data-current-clip','');await expect(canvas).not.toHaveAttribute('data-tracking-frame','');
 await page.getByLabel('Speech text').fill('Hello. It is lovely to meet you.');
 await page.getByRole('button',{name:'Generate & speak'}).click();
 await expect(page.locator('audio')).toHaveAttribute('src',/blob:/,{timeout:120000});
 await expect.poll(()=>page.locator('audio').evaluate((a:HTMLAudioElement)=>a.currentTime)).toBeGreaterThan(.2);
 await page.screenshot({path:info.outputPath('speech-in-progress.png'),fullPage:true});
 await expect.poll(async()=>Number(await canvas.getAttribute('data-face-motion'))).toBeGreaterThan(.05);
 const first=await canvas.getAttribute('data-current-clip');
 await expect(canvas).not.toHaveAttribute('data-current-clip',first!,{timeout:15000});
 await expect.poll(async()=>Number(await canvas.getAttribute('data-transitions')),{timeout:15000}).toBeGreaterThanOrEqual(2);
 await page.getByLabel('Show mesh').check();
 await page.screenshot({path:info.outputPath('speaking-avatar.png'),fullPage:true});
 await page.getByLabel('Replace face').uncheck();await page.getByLabel('Replace face').check();
 await page.getByRole('button',{name:'■ Stop'}).click();
 let trackingRequests=0;page.on('request',r=>{if(r.url().includes('/vision-wasm/')||r.url().includes('face_landmarker.task'))trackingRequests++});
 await page.reload();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:20000});
 await expect(page.getByLabel('Speech text')).toHaveValue('Hello. It is lovely to meet you.');
 expect(trackingRequests).toBe(0);
 await page.getByLabel('Face animation runtime').selectOption('webgpu');
 await page.getByRole('button',{name:'Generate & speak'}).click();
 await expect(page.locator('audio')).toHaveAttribute('src',/blob:/,{timeout:60000});
 await expect.poll(()=>page.locator('audio').evaluate((a:HTMLAudioElement)=>a.currentTime)).toBeGreaterThan(.2);
 await expect(page.getByRole('alert')).toHaveCount(0);
 expect(posts).toEqual([]);expect(errors).toEqual([]);
 const cache=await page.evaluate(async()=>{const db=await new Promise<IDBDatabase>((resolve,reject)=>{const r=indexedDB.open('loop-face-lab',1);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)});return await new Promise<any[]>((resolve,reject)=>{const r=db.transaction('records').objectStore('records').getAll();r.onsuccess=()=>resolve(r.result.filter(v=>v?.times).map(v=>({frames:v.frames.length,missing:v.missing,duration:v.duration})));r.onerror=()=>reject(r.error)})});
 await (await import('node:fs/promises')).writeFile(info.outputPath('verification.json'),JSON.stringify({status:'passed',cache,posts,pageErrors:errors},null,2));
 console.log(JSON.stringify({status:'passed',cache,posts,pageErrors:errors}));
});


test('loads the source image and loops from disk, persists them, and rejects generation routes',async({page,request})=>{
 const library=await(await request.get('/api/v1/loop-sets')).json(),set=library.sets.find((s:any)=>s.config.prompt==='The character talks and moves their hands.');
 const image=await(await request.get(`/api/v1/loop-sets/${set.id}/image`)).body();
 const clip=await(await request.get(`/api/v1/jobs/${set.clips[0].job.id}/video`)).body();
 await page.goto('/?runtime=cpu');await expect(page.getByLabel('Source image')).toBeEnabled();
 await page.getByLabel('Source image').setInputFiles({name:'source.png',mimeType:'image/png',buffer:image});
 await page.getByLabel('Video loops').setInputFiles({name:'uploaded-loop.mp4',mimeType:'video/mp4',buffer:clip});
 await expect(page.getByRole('button',{name:'Prepare face tracking'})).toBeEnabled();await page.getByRole('button',{name:'Prepare face tracking'}).click();
 await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:60000});
 await expect(page.locator('.video-row')).toContainText('241 frames cached');
 await page.reload();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:15000});
 await expect(page.getByAltText('Original source face')).toBeVisible();await expect(page.locator('.video-row')).toContainText('uploaded-loop.mp4');
 expect((await request.post('/api/v1/loop-sets')).status()).toBe(405);
});

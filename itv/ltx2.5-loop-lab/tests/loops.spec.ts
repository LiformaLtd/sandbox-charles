import {test,expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';
const setId='a'.repeat(64);
const ids=Array.from({length:5},(_,i)=>`00000000-0000-4000-8000-${String(i+1).padStart(12,'0')}`);
function batch(ready:number){return {id:setId,status:ready===5?'completed':'generating',ready_count:ready,total:5,config:{prompt:'A calm repeating scene',width:512,height:512,frames:241},clips:ids.map((id,slot)=>({slot,job:{id,status:slot<ready?'completed':slot===ready?'running':'queued',request:{seed:100+slot},depends_on:slot?ids[slot-1]:null}}))};}

test('starts with one clip, adds new clips, joins without blank frames or repeats, and restores cached clips',async({page})=>{
 let available=0,posts=0,videoRequests=0;
 const red=await readFile(new URL('./fixtures/red.mp4',import.meta.url));
 const blue=await readFile(new URL('./fixtures/blue.mp4',import.meta.url));
 await page.route('**/api/**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path.endsWith('/health'))return route.fulfill({json:{model_ready:true,detail:'Ready'}});
  if(path==='/api/v1/loop-sets'&&route.request().method()==='GET')return route.fulfill({json:{sets:[]}});
  if(path==='/api/v1/loop-sets'&&route.request().method()==='POST'){
   posts++;const body=route.request().postDataBuffer()!.toString();expect(body).toContain('A calm repeating scene');return route.fulfill({status:202,json:batch(available)});
  }
  if(path.endsWith('/video')){videoRequests++;const i=ids.findIndex(id=>path.includes(id));return route.fulfill({body:i%2?blue:red,contentType:'video/mp4'});}
  return route.fulfill({json:batch(available)});
 });
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await expect(page.getByLabel('Loop image')).toBeEnabled();
 await page.getByLabel('Loop image').setInputFiles(new URL('./fixtures/start.png',import.meta.url).pathname);
 await page.getByLabel('What happens inside the loop?').fill('A calm repeating scene');
 await page.getByRole('button',{name:'Build & start looping'}).click();
 await expect(page.getByText('Your first loop is on its way.')).toBeVisible();
 available=1;
 await expect(page.locator('canvas')).toHaveAttribute('data-current-clip',ids[0],{timeout:15000});
 await expect(page.getByText('Looping one clip while the others generate')).toBeVisible();
 available=3;
 await expect(page.locator('.count')).toContainText(/3\s*\/\s*5 ready/,{timeout:15000});
 await page.getByRole('button',{name:'♪ Audio off'}).click();
 await expect(page.getByRole('button',{name:'♪ Audio on'})).toHaveAttribute('aria-pressed','true');
 await page.getByLabel('Volume').press('Home');await page.getByLabel('Volume').press('ArrowRight');
 await page.getByRole('button',{name:'♪ Audio on'}).click();
 const samples=await page.evaluate(async()=>{
  const canvas=document.querySelector('canvas')!;const ctx=canvas.getContext('2d')!;
  const result:{black:number;frames:number;history:{id:string;at:number}[]}={black:0,frames:0,history:[]};
  const start=performance.now();let previous='';
  return await new Promise<typeof result>((resolve,reject)=>{
   function sample(){
    const id=canvas.dataset.currentClip??'';
    if(id&&id!==previous){result.history.push({id,at:performance.now()-start});previous=id;}
    const pixel=ctx.getImageData(32,32,1,1).data;if(pixel[0]+pixel[1]+pixel[2]<10)result.black++;result.frames++;
    if(Number(canvas.dataset.transitions)>=2)return resolve(result);
    if(performance.now()-start>30000)return reject(new Error('Player did not switch clips twice'));
    requestAnimationFrame(sample);
   }requestAnimationFrame(sample);
  });
 });
 expect(samples.black).toBe(0);expect(samples.frames).toBeGreaterThan(30);
 expect(samples.history.length).toBeGreaterThanOrEqual(3);
 for(let i=1;i<samples.history.length;i++)expect(samples.history[i].id).not.toBe(samples.history[i-1].id);
 await page.getByRole('button',{name:/Stop/}).click();
 const stopped=await page.getByLabel('Current loop progress').evaluate((p:HTMLProgressElement)=>p.value);
 await page.waitForTimeout(500);
 expect(await page.getByLabel('Current loop progress').evaluate((p:HTMLProgressElement)=>p.value)).toBe(stopped);
 expect(videoRequests).toBe(3);
 await page.reload();await expect(page.locator('.count')).toContainText(/3\s*\/\s*5 ready/);
 await expect(page.getByLabel('What happens inside the loop?')).toHaveValue('A calm repeating scene');
 expect(posts).toBe(1);expect(videoRequests).toBe(3);
 await page.getByRole('button',{name:/Start$/}).click();
 await expect(page.locator('canvas')).toHaveAttribute('data-current-clip',ids[0]);
 const pending=page.waitForEvent('download');await page.getByRole('link',{name:'Download loop 1'}).click();
 expect((await pending).suggestedFilename()).toContain('.mp4');
 expect(errors).toEqual([]);
});

test('recovers server-saved image, prompt and completed clips on a fresh browser origin',async({page})=>{
 const video=await readFile(new URL('./fixtures/red.mp4',import.meta.url));
 const image=await readFile(new URL('./fixtures/start.png',import.meta.url));let posts=0;
 await page.route('**/api/**',route=>{
  const path=new URL(route.request().url()).pathname;
  if(route.request().method()==='POST')posts++;
  if(path.endsWith('/health'))return route.fulfill({json:{model_ready:true,detail:'Ready'}});
  if(path==='/api/v1/loop-sets')return route.fulfill({json:{sets:[batch(5)]}});
  if(path.endsWith('/image'))return route.fulfill({body:image,contentType:'image/png'});
  if(path.endsWith('/video'))return route.fulfill({body:video,contentType:'video/mp4'});
  return route.fulfill({json:batch(5)});
 });
 await page.goto('/');
 await expect(page.getByLabel('What happens inside the loop?')).toHaveValue('A calm repeating scene');
 await expect(page.getByAltText('Shared first and last frame')).toBeVisible();
 await expect(page.locator('.count')).toContainText(/5\s*\/\s*5 ready/);
 expect(posts).toBe(0);
 await page.getByRole('button',{name:/Start$/}).click();
 await expect(page.locator('canvas')).toHaveAttribute('data-current-clip',ids[0]);
});

test('recovers invalid cached set metadata without clearing the draft or cached videos',async({page})=>{
 const video=await readFile(new URL('./fixtures/red.mp4',import.meta.url));
 const image=await readFile(new URL('./fixtures/start.png',import.meta.url));
 await page.route('**/api/**',route=>{
  const path=new URL(route.request().url()).pathname;
  if(path.endsWith('/health'))return route.fulfill({json:{model_ready:true,detail:'Ready'}});
  if(route.request().method()==='POST')return route.fulfill({status:202,json:{unexpected:'invalid set response'}});
  if(path==='/api/v1/loop-sets')return route.fulfill({json:{sets:[batch(5)]}});
  if(path.endsWith('/image'))return route.fulfill({body:image,contentType:'image/png'});
  if(path.endsWith('/video'))return route.fulfill({body:video,contentType:'video/mp4'});
  return route.fulfill({json:batch(5)});
 });
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('.count')).toContainText(/5\s*\/\s*5 ready/);
 await expect(page.getByLabel('What happens inside the loop?')).toBeEnabled();
 await page.getByLabel('What happens inside the loop?').fill('Keep my unsubmitted draft');
 await expect(page.getByText('Draft saved on this browser')).toBeVisible();
 await page.evaluate(async()=>{
  const db=await new Promise<IDBDatabase>((resolve,reject)=>{const r=indexedDB.open('ltx25-loop-lab',1);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)});
  await new Promise<void>((resolve,reject)=>{const tx=db.transaction('records','readwrite');tx.objectStore('records').put({unexpected:'old invalid metadata'},'active');tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error)});
 });
 await page.reload();
 await expect(page.getByLabel('What happens inside the loop?')).toHaveValue('Keep my unsubmitted draft');
 await expect(page.locator('.count')).toContainText(/5\s*\/\s*5 ready/);
 await expect(page.getByRole('alert')).toContainText('Recovered your saved clips');
 await page.getByRole('button',{name:'Build & start looping'}).click();
 await expect(page.getByRole('alert')).toContainText('invalid loop-set metadata');
 await expect(page.locator('.count')).toContainText(/5\s*\/\s*5 ready/);
 await page.reload();
 await expect(page.getByLabel('What happens inside the loop?')).toHaveValue('Keep my unsubmitted draft');
 await expect(page.locator('.count')).toContainText(/5\s*\/\s*5 ready/);
 expect(errors).toEqual([]);
});

test('switches saved sets without uploading or generating, restores selection, and opens cached sets offline',async({page})=>{
 const image=await readFile(new URL('./fixtures/start.png',import.meta.url));
 const red=await readFile(new URL('./fixtures/red.mp4',import.meta.url));
 const blue=await readFile(new URL('./fixtures/blue.mp4',import.meta.url));
 const first=batch(5);
 const second={...batch(5),id:'b'.repeat(64),config:{...first.config,prompt:'A second saved scene',width:768,height:512},
  clips:first.clips.map(c=>({...c,job:{...c.job,id:c.job.id.replace('00000000-','11111111-')}}))};
 let posts=0,offline=false,videoRequests=0;
 await page.route('**/api/**',route=>{
  const path=new URL(route.request().url()).pathname;
  if(route.request().method()==='POST')posts++;
  if(offline)return route.fulfill({status:503,json:{detail:'Test server offline'}});
  if(path.endsWith('/health'))return route.fulfill({json:{model_ready:true,detail:'Ready'}});
  if(path==='/api/v1/loop-sets')return route.fulfill({json:{sets:[first,second]}});
  if(path.endsWith('/image'))return route.fulfill({body:image,contentType:'image/png'});
  if(path.endsWith('/video')){videoRequests++;return route.fulfill({body:path.includes('11111111-')?blue:red,contentType:'video/mp4'});}
  return route.fulfill({json:path.includes(second.id)?second:first});
 });
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await expect(page.getByRole('button',{name:'Open saved set'})).toBeEnabled();
 await expect(page.locator('.count')).toContainText(/5\s*\/\s*5 ready/);
 await page.getByRole('button',{name:/Start$/}).click();
 await expect(page.locator('canvas')).toHaveAttribute('data-current-clip',ids[0]);
 await page.getByLabel('Saved loop sets',{exact:true}).selectOption(second.id);
 await page.getByRole('button',{name:'Open saved set'}).click();
 await expect(page.getByLabel('What happens inside the loop?')).toHaveValue(second.config.prompt);
 await expect(page.getByLabel('Format')).toHaveValue('768x512');
 await expect(page.locator('canvas')).toHaveAttribute('data-current-clip',second.clips[0].job.id);
 await expect(page.locator('.count')).toContainText(/5\s*\/\s*5 ready/);
 await expect(page.getByAltText('Shared first and last frame')).toBeVisible();
 expect(posts).toBe(0);expect(videoRequests).toBe(10);
 await page.reload();
 await expect(page.getByLabel('Saved loop sets',{exact:true})).toHaveValue(second.id);
 await expect(page.getByLabel('What happens inside the loop?')).toHaveValue(second.config.prompt);
 await expect(page.locator('.count')).toContainText(/5\s*\/\s*5 ready/);
 offline=true;await page.reload();
 await expect(page.getByRole('button',{name:'Open saved set'})).toBeEnabled();
 await page.getByLabel('Saved loop sets',{exact:true}).selectOption(first.id);
 await page.getByRole('button',{name:'Open saved set'}).click();
 await expect(page.getByLabel('What happens inside the loop?')).toHaveValue(first.config.prompt);
 await expect(page.getByLabel('Format')).toHaveValue('512x512');
 await expect(page.locator('.count')).toContainText(/5\s*\/\s*5 ready/);
 await page.getByRole('button',{name:/Start$/}).click();
 await expect(page.locator('canvas')).toHaveAttribute('data-current-clip',ids[0]);
 expect(posts).toBe(0);expect(videoRequests).toBe(10);expect(errors).toEqual([]);
});

test('shows cancelled sets, resumes unfinished clips, and reports replacement on a new build',async({page})=>{
 const video=await readFile(new URL('./fixtures/red.mp4',import.meta.url)),image=await readFile(new URL('./fixtures/start.png',import.meta.url));
 const stopped={...batch(1),status:'cancelled',clips:batch(1).clips.map(c=>({...c,job:{...c.job,status:c.slot===0?'completed':'cancelled'}}))};
 let current:any=stopped,retries=0,builds=0,videoRequests=0;
 await page.route('**/api/**',route=>{
  const path=new URL(route.request().url()).pathname;
  if(path.endsWith('/health'))return route.fulfill({json:{model_ready:true,detail:'Ready'}});
  if(path.endsWith('/retry')){retries++;current=batch(1);return route.fulfill({json:current});}
  if(path==='/api/v1/loop-sets'&&route.request().method()==='POST'){
   builds++;current={...batch(0),id:'b'.repeat(64),replaced_jobs:4,config:{...batch(0).config,prompt:'New motion immediately'}};
   return route.fulfill({status:202,json:current});
  }
  if(path==='/api/v1/loop-sets')return route.fulfill({json:{sets:[current]}});
  if(path.endsWith('/image'))return route.fulfill({body:image,contentType:'image/png'});
  if(path.endsWith('/video')){videoRequests++;return route.fulfill({body:video,contentType:'video/mp4'});}
  return route.fulfill({json:current});
 });
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.getByRole('button',{name:'Resume unfinished clips'})).toBeVisible();
 await expect(page.locator('.count')).toContainText(/1\s*\/\s*5 ready/);
 await expect(page.getByText('Cancelled by newer build')).toHaveCount(4);
 await page.getByRole('button',{name:'Resume unfinished clips'}).click();
 await expect(page.getByText('Generating…',{exact:true})).toBeVisible();expect(retries).toBe(1);expect(videoRequests).toBe(1);
 await page.getByLabel('What happens inside the loop?').fill('New motion immediately');
 await page.getByRole('button',{name:'Build & start looping'}).click();
 await expect(page.getByRole('status')).toContainText('Stopped 4 older unfinished jobs');
 await expect(page.getByText('Your first loop is on its way.')).toBeVisible();expect(builds).toBe(1);expect(errors).toEqual([]);
});

test('negative prompts have a default, reach submissions, persist, and restore with saved sets including blank legacy settings',async({page})=>{
 const image=await readFile(new URL('./fixtures/start.png',import.meta.url));
 const suggested='speaking, talking, lip movements, open mouth, singing',custom='talking, singing, teeth showing';
 const guided={...batch(0),id:'c'.repeat(64),config:{...batch(0).config,negative_prompt:custom,nag:true}};
 const plain={...batch(0),id:'d'.repeat(64)};
 let current=guided,submitted:string[]=[];
 await page.route('**/api/**',route=>{
  const path=new URL(route.request().url()).pathname;
  if(path.endsWith('/health'))return route.fulfill({json:{model_ready:true,detail:'Ready'}});
  if(path==='/api/v1/loop-sets'&&route.request().method()==='POST'){
   const body=route.request().postDataBuffer()!.toString(),negative=body.match(/name="negative_prompt"\r\n\r\n([\s\S]*?)\r\n--/)?.[1];
   expect(negative).toBeDefined();submitted.push(negative!);current=(negative?guided:plain) as typeof guided;
   return route.fulfill({status:202,json:current});
  }
  if(path==='/api/v1/loop-sets')return route.fulfill({json:{sets:submitted.length?[guided,plain]:[]}});
  if(path.endsWith('/image'))return route.fulfill({body:image,contentType:'image/png'});
  return route.fulfill({json:path.includes(guided.id)?guided:path.includes(plain.id)?plain:current});
 });
 await page.goto('/');const field=page.getByLabel('Negative prompt',{exact:false});
 await expect(field).toHaveValue(suggested);
 await expect(page.getByLabel('Loop image')).toBeEnabled();
 await page.getByLabel('Loop image').setInputFiles(new URL('./fixtures/start.png',import.meta.url).pathname);
 await page.getByLabel('What happens inside the loop?').fill('A calm repeating scene');await field.fill(custom);
 await page.getByRole('button',{name:'Build & start looping'}).click();await expect(page.getByText('Your first loop is on its way.')).toBeVisible();
 expect(submitted).toEqual([custom]);await page.reload();await expect(field).toHaveValue(custom);
 await field.fill('');await page.getByRole('button',{name:'Build & start looping'}).click();await expect(page.getByRole('button',{name:'Build & start looping'})).toBeEnabled();
 expect(submitted).toEqual([custom,'']);await page.reload();await expect(field).toHaveValue('');
 await page.getByLabel('Saved loop sets',{exact:true}).selectOption(guided.id);await page.getByRole('button',{name:'Open saved set'}).click();await expect(field).toHaveValue(custom);
 await page.getByLabel('Saved loop sets',{exact:true}).selectOption(plain.id);await page.getByRole('button',{name:'Open saved set'}).click();await expect(field).toHaveValue('');
});

import {test,expect} from '@playwright/test';

test('male and female voices generate different real speech, reuse the model and restore the selected voice',async({page,context},info)=>{
 const errors:string[]=[],requests:string[]=[];page.on('pageerror',e=>errors.push(e.message));context.on('request',r=>requests.push(r.url()));
 await page.goto('/?runtime=cpu');await expect(page.getByLabel('Phonon voice',{exact:true})).toBeEnabled();
 const picker=page.getByLabel('Phonon voice',{exact:true});await expect(picker.locator('optgroup[label="Female voices"] option')).toHaveCount(4);await expect(picker.locator('optgroup[label="Male voices"] option')).toHaveCount(4);
 for(const id of['zoey','juniper','freya','maeve','easton','trevor','alfie','hugh'])expect((await page.request.get(`/assets/lf-tts/${id}.safetensors`)).status()).toBe(200);
 const library=await(await page.request.get('/api/v1/loop-sets')).json(),talking=library.sets.find((s:any)=>s.config.prompt==='The character talks and moves their hands.');
 await page.getByLabel('Saved Loop Lab sets').selectOption(talking.id);for(const n of[2,3,4,5])await page.getByRole('checkbox',{name:`Loop ${n}`,exact:true}).uncheck();
 await page.getByRole('button',{name:'Load selected loops & source image'}).click();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:90000});
 await picker.selectOption('freya');
 await expect.poll(()=>page.evaluate(async()=>{
  const {read}=await import(/* @vite-ignore */ '/src/lib/cache.ts');return read('speech-voice');
 })).toBe('freya');
 await page.reload();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:20000});await expect(picker).toHaveValue('freya');
 await page.getByLabel('Speech text').fill('Hello. It is lovely to meet you.');await page.getByRole('button',{name:'Generate & speak'}).click();
 await expect(page.locator('.viewer .hint').filter({hasText:'Freya · UK'})).toBeVisible({timeout:120000});
 await expect(page.getByLabel('Generated speech')).toHaveAttribute('src',/blob:/);
 const audioHash=()=>page.evaluate(async()=>{
  const audio=document.querySelector('audio')!,bytes=await(await fetch(audio.src)).arrayBuffer(),hash=await crypto.subtle.digest('SHA-256',bytes);
  return {sha256:Array.from(new Uint8Array(hash),v=>v.toString(16).padStart(2,'0')).join(''),bytes:bytes.byteLength};
 });
 const female=await audioHash();expect(female.bytes).toBeGreaterThan(10000);
 await expect(picker).toBeEnabled();await picker.selectOption('hugh');await expect(page.getByLabel('Generated speech')).not.toHaveAttribute('src',/blob:/);
 await page.getByRole('button',{name:'Generate & speak'}).click();await expect(page.locator('.viewer .hint').filter({hasText:'Hugh · UK'})).toBeVisible({timeout:60000});
 const male=await audioHash();expect(male.bytes).toBeGreaterThan(10000);expect(male.sha256).not.toBe(female.sha256);
 expect(requests.some(url=>url.includes('/freya.safetensors'))).toBe(true);expect(requests.some(url=>url.includes('/hugh.safetensors'))).toBe(true);
 expect(requests.filter(url=>url.includes('/model.q8.gguf'))).toHaveLength(1);
 await page.screenshot({path:info.outputPath('voice-picker.png'),fullPage:true});
 await page.reload();await expect(page.getByRole('button',{name:'Generate & speak'})).toBeEnabled({timeout:20000});await expect(picker).toHaveValue('hugh');expect(errors).toEqual([]);
 await (await import('node:fs/promises')).writeFile(info.outputPath('voice-selection.json'),JSON.stringify({female:{voice:'Freya',...female},male:{voice:'Hugh',...male},voiceCount:8,modelLoads:1,restored:'hugh',pageErrors:errors},null,2));
});

import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {webkit,devices}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await webkit.launch({executablePath:process.env.WEBKIT_EXECUTABLE});
try{
 const page=await browser.newPage({...devices['iPhone 13']});let navigations=0,missing=0;
 page.on('framenavigated',frame=>{if(frame===page.mainFrame())navigations++;});
 // Simulate assets removed by a deployment while the old entry is executing.
 await page.route('**/assets/*.js',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(navigations===1&&(path.includes('/onnx-tts-')||(path.includes('/index-')&&route.request().method()==='HEAD'))){
   missing++;await route.fulfill({status:404,contentType:'text/plain',body:'NOT_FOUND'});
  }else await route.continue();
 });
 await page.goto(process.env.DEPLOYMENT_URL);await page.waitForFunction(()=>!document.querySelector('#generate').disabled);
 const dialogue='Please keep this custom dialogue after the update.';
 await page.locator('#speech').fill(dialogue);await page.locator('input[name=engine][value=supertonic]').check();
 await page.locator('#head-motion').evaluate(el=>{el.value='42';el.dispatchEvent(new Event('input',{bubbles:true}));});
 await page.locator('#generate').click();
 await page.waitForURL(()=>navigations>=2,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(text=>document.querySelector('#speech')?.value===text&&!document.querySelector('#generate').disabled,dialogue);
 assert.equal(navigations,2);assert.ok(missing>0);
 assert.equal(await page.locator('input[name=engine]:checked').inputValue(),'supertonic');
 assert.equal(await page.locator('#head-motion').inputValue(),'42');
 assert.equal(await page.locator('select[name=animation]').inputValue(),'compact-v4');
 assert.equal(await page.locator('select[name=runtime]').inputValue(),'webgpu');
 const result={base:process.env.DEPLOYMENT_URL,navigations,missing,preservedDialogue:dialogue,preservedEngine:'supertonic',preservedMotion:42};
 await writeFile('.local/tts-deployment-fix/recovery.json',JSON.stringify(result,null,2));console.log(result);
}finally{await browser.close();}

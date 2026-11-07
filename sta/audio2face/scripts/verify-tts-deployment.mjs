import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {webkit,devices}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const base=process.env.DEPLOYMENT_URL;
if(!base)throw Error('Set DEPLOYMENT_URL');
const results=[];
for(const engine of (process.env.TTS_ENGINES||'supertonic,pocket').split(',')){
 const browser=await webkit.launch(process.env.WEBKIT_EXECUTABLE?{executablePath:process.env.WEBKIT_EXECUTABLE}:{});
 try{
  const page=await browser.newPage({...devices['iPhone 13']});const errors=[],failed=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)failed.push({url:r.url(),status:r.status()});});
  await page.addInitScript(()=>{let starts=0;const start=AudioBufferSourceNode.prototype.start;AudioBufferSourceNode.prototype.start=function(...a){starts++;return start.apply(this,a);};window.audioStarts=()=>starts;});
  await page.goto(base);await page.waitForFunction(()=>!document.querySelector('#generate').disabled);
  await page.locator(`input[name=engine][value=${engine}]`).check();await page.locator('#speech').fill('Please bring my blue bag.');
  await page.locator('#generate').click();
  await page.waitForFunction(()=>document.querySelector('#status').textContent==='Animation ready to replay.'||document.querySelector('#status').classList.contains('error'),null,{timeout:240000});
  const status=await page.locator('#status').textContent(),audio=await page.evaluate(()=>window.audioStarts());
  const result={engine,status,audio,errors,failed};results.push(result);console.log(JSON.stringify(result));
  assert.equal(status,'Animation ready to replay.');assert.ok(audio>0);assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
 }finally{await browser.close();await writeFile(process.env.TTS_VALIDATION_OUTPUT||'.local/tts-deployment-fix/smoke.json',JSON.stringify({base,results},null,2));}
}

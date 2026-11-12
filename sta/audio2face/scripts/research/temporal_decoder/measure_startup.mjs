import {writeFile} from 'node:fs/promises';
const {webkit,devices}=await import(process.env.PLAYWRIGHT_MODULE);
const base=process.env.DIAGNOSTIC_URL,root=process.env.A2F_TEMPORAL_OUTPUT;
const models={v4:['/.local/compact-coverage-v2/coverage/runs/control-replay-s37/model.onnx','/src/compact-webgpu.js'],control:['/.local/compact-temporal-v1/control/model.onnx','/.local/compact-temporal-v1/browser/control-runner.js'],temporal:['/.local/compact-temporal-v1/temporal/model.onnx','/.local/compact-temporal-v1/browser/temporal-runner.js']};
const rows=[];
for(let repeat=0;repeat<3;repeat++)for(const [model,[modelUrl,runnerUrl]] of Object.entries(models)){
 const browser=await webkit.launch({executablePath:process.env.WEBKIT_EXECUTABLE});
 try{
  const page=await browser.newPage({...devices['iPhone 13']});await page.goto(base);
  const values=await page.evaluate(async({modelUrl,runnerUrl})=>{
   const {createCompactWebGpu}=await import(runnerUrl);
   const adapter=await navigator.gpu.requestAdapter();
   const at=performance.now(),data=await(await fetch(modelUrl)).arrayBuffer(),loaded=performance.now();
   const run=await createCompactWebGpu(adapter,data,()=>{});const ready=performance.now();
   run.dispose?.();return {local_fetch_ms:loaded-at,session_create_ms:ready-loaded};
  },{modelUrl,runnerUrl});rows.push({repeat,model,...values});
 }finally{await browser.close();}
}
await writeFile(root+'/browser/startup.json',JSON.stringify({method:'Fresh WebKit per model, cached local assets and previously compiled shaders; excludes rig/TTS and first-ever CDN download',rows},null,2));console.log(rows);

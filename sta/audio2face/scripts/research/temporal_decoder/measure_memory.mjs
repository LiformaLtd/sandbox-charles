// Isolated STA inference with the normal animated James page, no TTS session.
import {execFileSync,spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {writeFile,mkdir} from 'node:fs/promises';
const {webkit,devices}=await import(process.env.PLAYWRIGHT_MODULE);
const base=process.env.DIAGNOSTIC_URL,root=process.env.A2F_TEMPORAL_OUTPUT;
if(!base||!root||!process.env.A2F_MEMORY_SAMPLER)throw Error('Set diagnostic URL, output root and compiled sampler.');
const processes=()=>execFileSync('/bin/ps',['-axo','pid=,comm='],{encoding:'utf8'}).split('\n').map(x=>x.trim().match(/^(\d+)\s+(.*)$/)).filter(Boolean).map(([,id,cmd])=>({id:Number(id),cmd}));
const models={v4:['/.local/compact-coverage-v2/coverage/runs/control-replay-s37/model.onnx','/src/compact-webgpu.js'],control:['/.local/compact-temporal-v1/control/model.onnx','/.local/compact-temporal-v1/browser/control-runner.js'],temporal:['/.local/compact-temporal-v1/temporal/model.onnx','/.local/compact-temporal-v1/browser/temporal-runner.js']};
await mkdir(root+'/memory',{recursive:true});
for(const [name,[modelUrl,runnerUrl]] of Object.entries(models)){
 const previous=new Set(processes().map(p=>p.id));
 const browser=await webkit.launch({executablePath:process.env.WEBKIT_EXECUTABLE});let sampling;
 try{
  const page=await browser.newPage({...devices['iPhone 13']});const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/?model=compact-v4&sta=webgpu');
  await page.waitForFunction(()=>document.querySelector('canvas')&&!document.querySelector('#generate').disabled,null,{timeout:60000});
  await page.waitForTimeout(1200);
  const pids=processes().filter(p=>!previous.has(p.id)&&/WebKit.*(WebContent|GPU)/.test(p.cmd));
  if(!pids.some(p=>p.cmd.includes('WebContent'))||!pids.some(p=>p.cmd.includes('GPU')))throw Error('Missing renderer/GPU process');
  const samples=[],stages=[];
  sampling=spawn(process.env.A2F_MEMORY_SAMPLER,pids.map(p=>String(p.id)));
  createInterface({input:sampling.stdout}).on('line',line=>{
   const [timestamp,...values]=line.trim().split(' ');const parts=values.map(v=>v.split(':').map(Number));
   if(parts.every(p=>p[1]>=0))samples.push({timestamp:Number(timestamp),processes:parts,bytes:parts.reduce((n,p)=>n+p[1],0)});
  });
  const mark=async stage=>{await page.waitForTimeout(250);stages.push({name:stage,at:Date.now()});};
  await mark('idle');
  await page.evaluate(async({modelUrl,runnerUrl})=>{
   const original=GPUDevice.prototype.createBuffer;let bytes=0,buffers=0;
   GPUDevice.prototype.createBuffer=function(d){const b=original.call(this,d);bytes+=b.size;buffers++;return b;};
   window.modelAllocation=()=>({bytes,buffers});
   const {createCompactWebGpu}=await import(runnerUrl);
   const model=await(await fetch(modelUrl)).arrayBuffer();
   window.memoryRun=await createCompactWebGpu(await navigator.gpu.requestAdapter(),model,()=>{});
  },{modelUrl,runnerUrl});
  await mark('ready');
  for(const [stage,file,repeats] of [['short','phonon-original.f32',1],['long','phonon-packed.f32',1],['long-repeat','phonon-packed.f32',3]]){
   await page.evaluate(async({file,repeats})=>{
    const {audioWindows}=await import('/src/geometry.mjs');
    const audio=new Float32Array(await(await fetch('/.local/compact-temporal-v1/browser/parity/'+file)).arrayBuffer());
    for(let repeat=0;repeat<repeats;repeat++){
     window.memoryRun.reset?.();
     async function* chunks(){for(let i=0;i<audio.length;i+=1280)yield audio.slice(i,i+1280);}
     for await(const block of audioWindows(chunks())){
      if(!block.sampleIndices.length)continue;
      const values=await window.memoryRun(block.samples,{first:block.first,count:block.sampleIndices.length});
      if(!values.every(Number.isFinite))throw Error('Invalid memory-test track');
     }
    }
   },{file,repeats});
   await mark(stage);
  }
  await page.waitForTimeout(1000);await mark('settled');
  const allocation=await page.evaluate(()=>window.modelAllocation());
  sampling.kill();sampling=undefined;
  const peak=Math.max(...samples.map(s=>s.bytes));
  if(errors.length)throw Error(errors.join(';'));
  await writeFile(root+'/memory/'+name+'.json',JSON.stringify({name,modelUrl,pids,stages,samples,peak,allocation,errors,method:'20ms proc_pid_rusage physical footprint, renderer+GPU concurrent sum; normal James idle; prerecorded STA, no TTS'},null,2));
  console.log(name,'peak MiB',(peak/2**20).toFixed(1),allocation);
 }finally{sampling?.kill();await browser.close();}
}

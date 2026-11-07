// Repeatable local browser benchmark; browser pages contain only the frozen demo copy.
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';
const { webkit, devices } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
import { execFileSync,spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { readFile, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const root=pathToFileURL(join(process.env.A2F_ABLATION_OUTPUT ?? '.local/compact-ablation','/'));
const work=process.env.A2F_RESEARCH_WORK;
const base=process.env.BENCH_URL;
if(!base || !work)throw Error('Set BENCH_URL, A2F_RESEARCH_WORK and A2F_ABLATION_OUTPUT.');
const processInfo=()=>execFileSync('/bin/ps',['-axo','pid=,comm='],{encoding:'utf8'}).split('\n').map(line=>line.trim().match(/^(\d+)\s+(.*)$/)).filter(Boolean).map(([,id,cmd])=>({id:Number(id),cmd}));
const launch=()=>webkit.launch(process.env.WEBKIT_EXECUTABLE?{executablePath:process.env.WEBKIT_EXECUTABLE}:{});
const ua='Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1';
const fixture=JSON.parse(await readFile(join(work,'arkit-student-v2/manifest.json'))).find(x=>x.stem==='test-phonon-v2-267');
for(const candidate of process.argv.slice(2)){
  // Parity uses a separate browser so it cannot warm the measured session.
  if(candidate!=='baseline'){
    const browser=await launch();
    try{
      const page=await browser.newPage({...devices['iPhone 13'],userAgent:ua});
      await page.goto(base+'/parity.html?candidate='+candidate);
      await page.waitForFunction(()=>['Passed','Failed'].includes(document.querySelector('#status').textContent),null,{timeout:60000});
      const status=await page.locator('#status').textContent(),result=await page.locator('#result').textContent();
      assert.equal(status,'Passed',result);await writeFile(new URL(candidate+'-browser-parity.json',root),result);console.log(candidate,'parity',result);
    }finally{await browser.close();}
  }
  if(process.env.PARITY_ONLY==='1')continue;
  for(const length of (process.env.SMOKE==='1'?['short']:['short','long']))for(let repeat=1;repeat<=(process.env.SMOKE==='1'?1:Number(process.env.BENCH_REPEATS??2));repeat++){
    const previous=new Set(processInfo().map(p=>p.id));const browser=await launch();let sampling;
    const variant=`${candidate}-${length}-${repeat}`;
    try{
      const page=await browser.newPage({...devices['iPhone 13'],userAgent:ua});
      await page.addInitScript(()=>{
        let poses=0,firstAudio,clicked,live=0,peak=0,buffers=0,pipelines=0;
        const sizes=new WeakMap();
        const update=WebGL2RenderingContext.prototype.bufferSubData;
        WebGL2RenderingContext.prototype.bufferSubData=function(...args){poses++;return update.apply(this,args);};
        const create=GPUDevice.prototype.createBuffer,destroy=GPUBuffer.prototype.destroy;
        GPUDevice.prototype.createBuffer=function(...args){const b=create.apply(this,args);sizes.set(b,b.size);live+=b.size;peak=Math.max(peak,live);buffers++;return b;};
        GPUBuffer.prototype.destroy=function(){live-=sizes.get(this)??0;sizes.delete(this);return destroy.call(this);};
        for(const key of ['createComputePipeline','createComputePipelineAsync']){const fn=GPUDevice.prototype[key];GPUDevice.prototype[key]=function(...args){pipelines++;return fn.apply(this,args);};}
        const start=AudioBufferSourceNode.prototype.start;
        AudioBufferSourceNode.prototype.start=function(...args){firstAudio??=Date.now();return start.apply(this,args);};
        document.addEventListener('click',e=>{if(e.target.closest('#generate'))clicked=Date.now();},true);
        window.staMs=[];window.stats=()=>({poses,firstAudio,clicked,gpuLive:live,gpuPeak:peak,buffers,pipelines,staMs:[...window.staMs]});
      });
      let navigations=0;page.on('framenavigated',frame=>{if(frame===page.mainFrame())navigations++;});
      const errors=[],consoleMessages=[];page.on('pageerror',error=>errors.push(error.message));
      page.on('console',message=>{if(['warning','error'].includes(message.type()))consoleMessages.push({type:message.type(),text:message.text()});});
      await page.goto(base+'/?model=compact-v3&sta=webgpu&candidate='+candidate);
      const ready=()=>page.waitForFunction(()=>!document.querySelector('#generate').disabled,null,{timeout:60000});await ready();
      const pids=processInfo().filter(p=>!previous.has(p.id)&&/WebKit.*(WebContent|GPU)/.test(p.cmd)).map(p=>({...p,type:p.cmd.includes('WebContent')?'renderer':'GPU'}));
      assert.equal(pids.length,2,JSON.stringify(pids));
      const samples=[],stages=[];
      sampling=spawn(new URL('sample',root).pathname,pids.map(p=>String(p.id)));
      createInterface({input:sampling.stdout}).on('line',line=>{const [ts,...data]=line.split(' ');samples.push({ts:Number(ts),processes:data.map(v=>v.split(':').map(Number))});});
      const mark=async name=>{const s=await page.evaluate(()=>({at:Date.now(),status:document.querySelector('#status').textContent,duration:document.querySelector('#duration').textContent,...stats()}));stages.push({name,...s});};
      await mark('idle');
      await page.locator('#prepare-tts').click();await page.waitForFunction(()=>document.querySelector('#status').textContent==='Phonon is ready.'||document.querySelector('#status').classList.contains('error'),null,{timeout:60000});assert.equal(await page.locator('#status').textContent(),'Phonon is ready.');await ready();await mark('tts');
      await page.locator('#prepare-sta').click();await page.waitForFunction(()=>document.querySelector('#status').textContent==='STA is ready.'||document.querySelector('#status').classList.contains('error'),null,{timeout:60000});assert.equal(await page.locator('#status').textContent(),'STA is ready.');await ready();await mark('both');
      await page.locator('#speech').fill(length==='short'?fixture.text:'Meet me in the morning. We can make a plan together. After breakfast, we will pack the blue bags, check the map, and walk down to the river. The path goes past the old bridge, where we can stop and watch the water before heading home.');
      await page.evaluate(()=>{window.lastRenderedFrame=-1;new MutationObserver(()=>{const frame=Number(document.querySelector('#viewport').dataset.frame);if(Number.isFinite(frame))lastRenderedFrame=Math.max(lastRenderedFrame,frame);}).observe(document.querySelector('#viewport'),{attributes:true,attributeFilter:['data-frame']});});
      await mark('start');await page.locator('#generate').click();await page.waitForFunction(()=>document.querySelector('#status').textContent==='Animation ready to replay.'||document.querySelector('#status').classList.contains('error'),null,{timeout:60000});await ready();await mark('end');
      assert.equal(navigations,1,'Unexpected page reload during measurement');assert.equal(stages.at(-1).status,'Animation ready to replay.');assert.ok(stages.at(-1).firstAudio);assert.deepEqual(errors,[]);
      const lastFrame=await page.evaluate(()=>lastRenderedFrame);assert.ok(lastFrame>60);
      const poses=await page.evaluate(()=>stats().poses);await page.waitForFunction(n=>stats().poses>n+10,poses);await mark('idle-after');
      const peak=Math.max(...samples.map(s=>s.processes.reduce((n,p)=>n+Math.max(0,p[1]),0)))/1e6;
      const end=stages.find(s=>s.name==='end');const times=end.staMs.slice(1).sort((a,b)=>a-b);
      const summary={candidate,length,repeat,peak_MB:peak,block_ms:times[Math.floor(times.length/2)],first_block_ms:end.staMs[0],time_to_audio_ms:end.firstAudio-end.clicked,gpu_peak_bytes:end.gpuPeak,buffers:end.buffers,pipelines:end.pipelines,duration:end.duration,lastFrame,idle_after:true,navigations};
      await writeFile(new URL('browser-'+variant+'.json',root),JSON.stringify({summary,pids,stages,samples,consoleMessages},null,2));console.log(variant,summary);
    }finally{sampling?.kill();await browser.close();}
  }
}

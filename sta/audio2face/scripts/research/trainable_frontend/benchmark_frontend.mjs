// Node CPU microbenchmark of the existing JS frontend, not a browser/GPU benchmark.
import {readFile,writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const [modulePath,fixturePath,output]=process.argv.slice(2);
const {createLogMel}=await import(pathToFileURL(modulePath));
const fixtures=JSON.parse(await readFile(fixturePath,'utf8'));
const read=async path=>{const b=await readFile(path);return new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const window=await read(fixtures.window),pcm=await read(fixtures.pcm),reports={};
for(const row of fixtures.models){
 const mel=await read(row.mel),fn=createLogMel(window,mel);
 for(let i=0;i<15;i++)fn(pcm);
 const timings=[];
 for(let i=0;i<100;i++){const start=performance.now();const result=fn(pcm);timings.push(performance.now()-start);if(!result.every(Number.isFinite))throw Error('Nonfinite features');}
 timings.sort((a,b)=>a-b);reports[row.name]={medianMs:timings[50],p95Ms:timings[95],positiveWeights:mel.filter(v=>v>0).length};
}
await writeFile(output,JSON.stringify({runtime:process.version,note:'Sequential warm Node CPU microbenchmark, one training-audio window; excludes neural inference, browser overhead and GPU. Band density can affect existing JS summation cost.',models:reports},null,2));
console.log(JSON.stringify(reports));

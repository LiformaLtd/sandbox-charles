import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { windowFrames } from '../src/lib/windows.ts';
import { ShapeSampler, parseMapping, sampleTrack } from '../src/lib/shape.ts';
import { playbackWav } from '../src/lib/audio.ts';
import type { AvatarMesh, ContactRig } from '../src/lib/types.ts';

const read=(name:string)=>JSON.parse(readFileSync(new URL(`../static/assets/${name}`,import.meta.url),'utf8'));
for(const length of [1,266,267,7999,8000,8001,16000,16001,159999])test(`inference windows cover ${length} samples exactly once`,()=>{
 const seen=new Map<number,number>();
 for(let start=-16000;start<length;start+=8000)for(const frame of windowFrames(start,length))seen.set(frame.global,(seen.get(frame.global)??0)+1);
 assert.equal(seen.size,Math.ceil(length*60/16000));
 for(let f=0;f<Math.ceil(length*60/16000);f++)assert.equal(seen.get(f),1);
});
for(const kind of ['linear','local'])test(`${kind} mapping preserves the avatar reference and strength zero`,()=>{
 const catalog=read('catalog.json');
 for(const avatar of catalog.avatars){
  const mesh=read(`avatars/${avatar.id}/mesh.json`) as AvatarMesh;
  const map=parseMapping(read(`avatars/${avatar.id}/${kind}.json`));
  const sampler=new ShapeSampler(mesh,map,catalog.contact as ContactRig);
  const neutral=sampler.sample(map.sourceReference,1,true).slice();
  for(let i=0;i<478;i++){
   const x=map.reference[i*2],y=map.reference[i*2+1],r=mesh.frame.rotation;
   assert.ok(Math.abs(neutral[i*2]-((x*r[0]+y*r[1])*mesh.frame.width+mesh.frame.center[0]))<1e-6);
   assert.ok(Math.abs(neutral[i*2+1]-((x*r[2]+y*r[3])*mesh.frame.width+mesh.frame.center[1]))<1e-6);
  }
  const extreme=new Float32Array(52).fill(0.7);
  assert.deepEqual(sampler.sample(extreme,0,true),neutral);
  assert.ok(sampler.sample(extreme,1,true).every(Number.isFinite));
 }
});
test('audio clock interpolation clamps and does not normalize independent channels',()=>{
 const frames=new Float32Array(104);frames.fill(0.8,0,52);frames.fill(1,52);const out=new Float32Array(52);
 sampleTrack(frames,1/120,out);assert.ok(Math.abs(out[0]-0.9)<1e-6);assert.ok(out.reduce((a,b)=>a+b,0)>40);
 sampleTrack(frames,-1,out);assert.ok(Math.abs(out[0]-0.8)<1e-6);sampleTrack(frames,100,out);assert.equal(out[0],1);
});
test('inference PCM equals WAV playback samples including clipping',async()=>{
 const sound=playbackWav(Float32Array.of(-2,-0.5,0,0.5,2));const view=new DataView(await sound.blob.arrayBuffer());
 assert.equal(view.getUint32(24,true),16000);
 for(let i=0;i<sound.pcm.length;i++)assert.equal(sound.pcm[i],view.getInt16(44+i*2,true)/32768);
 assert.throws(()=>playbackWav(Float32Array.of(NaN)),/invalid/);
});

for(const kind of ['linear','local'])test(`${kind} cheek transfer moves cheeks while preserving reference, lips and upper face on every fit`,()=>{
 const catalog=read('catalog.json');
 const protectedPoints=[78,191,80,81,82,13,312,311,310,415,308,95,88,178,87,14,317,402,318,324,61,185,40,39,37,0,267,269,270,409,291,146,91,181,84,17,314,405,321,375,33,133,263,362,145,374,10];
 for(const avatar of catalog.avatars){
  const mesh=read(`avatars/${avatar.id}/mesh.json`) as AvatarMesh;
  const map=parseMapping(read(`avatars/${avatar.id}/${kind}.json`));const sampler=new ShapeSampler(mesh,map,catalog.contact);
  const neutral=sampler.sample(map.sourceReference,1,true,0).slice();
  assert.deepEqual(sampler.sample(map.sourceReference,1,true,1),neutral,avatar.id+' neutral');
  const weights=Float32Array.from(map.sourceReference,(_,i)=>i>=14&&i<=40?0.35:map.sourceReference[i]);
  const old=sampler.sample(weights,1,true,0).slice();const next=sampler.sample(weights,1,true,1).slice();
  for(const i of protectedPoints){assert.equal(next[i*2],old[i*2],`${avatar.id} protected x ${i}`);assert.equal(next[i*2+1],old[i*2+1],`${avatar.id} protected y ${i}`);}
  assert.ok([205,425].some(i=>Math.hypot(next[i*2]-old[i*2],next[i*2+1]-old[i*2+1])>1e-4),`${avatar.id} moving cheeks`);
  assert.ok(next.every(Number.isFinite));assert.deepEqual(sampler.sample(weights,0,true,1),neutral);
  const tri=mesh.triangles;const area=(p:Float32Array,a:number,b:number,c:number)=>(p[b*2]-p[a*2])*(p[c*2+1]-p[a*2+1])-(p[b*2+1]-p[a*2+1])*(p[c*2]-p[a*2]);
  for(let j=0;j<tri.length;j+=3){const [a,b,c]=tri.slice(j,j+3);if(Math.max(a,b,c)>=478)continue;
   const before=area(old,a,b,c),after=area(next,a,b,c);
   if(Math.abs(before)>1e-9)assert.ok(after*Math.sign(before)>=Math.abs(before)*.095-1e-8,`${avatar.id} cheek triangle ${j/3} maintains area`);
  }
 }
});

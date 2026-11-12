import {test} from 'node:test';import assert from 'node:assert/strict';
import {smoothYawTrack,smoothPitchTrack,yawRotation} from '../src/lib/rotation-smoothing.ts';
import {rotationAngles} from '../src/lib/pose.ts';
import type {PoseTransform} from '../src/lib/pose.ts';import type {FaceTrack} from '../src/lib/media-types.ts';
function pose(yaw:number,n=0):PoseTransform{return {rotation:yawRotation(5+n*.01,yaw,-2+n*.005),pitch:5+n*.01,yaw,roll:-2+n*.005,scale:94,x:256+n*.1,y:120+n*.2,centerX:0,centerY:0}}
function track(times:number[]):FaceTrack{return {times,frames:times.map(()=>null),duration:times.at(-1)!+1/24,width:512,height:512,missing:0,revision:1}}
const rms=(values:number[])=>Math.sqrt(values.reduce((sum,v)=>sum+v*v,0)/values.length);
test('yaw noise is strongly reduced without phase lag or loss of a gradual head turn',()=>{
 const times=Array.from({length:241},(_,i)=>i/24),t=track(times),clean=times.map(time=>10*Math.sin(time*2*Math.PI/10));
 const raw=clean.map((yaw,n)=>pose(yaw+(n%2?1.4:-1.4),n)),before=raw.map(p=>p.yaw),filtered=smoothYawTrack(t,raw,300,0);
 const indices=times.map((_,i)=>i).filter(i=>times[i]>.6&&times[i]<9.4);
 const original=rms(indices.map(i=>raw[i].yaw-clean[i])),after=rms(indices.map(i=>filtered[i]!.yaw-clean[i]));
 assert.ok(after<original*.15,`${original} → ${after}`);
 const peak=filtered.reduce((best,p,n)=>p!.yaw>filtered[best]!.yaw?n:best,0);
 assert.ok(Math.abs(times[peak]-2.5)<=1/24);
 assert.ok(Math.abs(filtered[peak]!.yaw-10)<.2);assert.deepEqual(raw.map(p=>p.yaw),before);
});
test('roll, pitch, translation and fixed scale remain unchanged and rotations remain proper',()=>{
 const times=Array.from({length:49},(_,i)=>i/24),raw=times.map((time,n)=>pose(4*Math.sin(time*8),n)),filtered=smoothYawTrack(track(times),raw,400,0);
 for(let i=0;i<raw.length;i++){
  const p=filtered[i]!,r=p.rotation,angles=rotationAngles(r);
  for(const key of ['pitch','roll','x','y','scale','centerX','centerY'] as const)assert.equal(p[key],raw[i][key]);
  assert.ok(Math.abs(angles.pitch-p.pitch)<.00001);assert.ok(Math.abs(angles.roll-p.roll)<.00001);
  for(let row=0;row<3;row++)assert.ok(Math.abs(Math.hypot(r[row*3],r[row*3+1],r[row*3+2])-1)<1e-6);
  const determinant=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);assert.ok(Math.abs(determinant-1)<1e-6);
 }
});
test('different same-image loops share endpoint yaw and ease into the join',()=>{
 const times=Array.from({length:241},(_,i)=>i/24),t=track(times),anchor=1.25;
 const a=smoothYawTrack(t,times.map((time,n)=>pose(3+Math.sin(time),n)),300,anchor),b=smoothYawTrack(t,times.map((time,n)=>pose(-2+Math.cos(time),n)),300,anchor);
 assert.equal(a[0]!.yaw,anchor);assert.equal(a.at(-1)!.yaw,anchor);assert.equal(b[0]!.yaw,anchor);assert.equal(b.at(-1)!.yaw,anchor);
 assert.ok(Math.abs(a[1]!.yaw-anchor)<.1);assert.ok(Math.abs(a.at(-2)!.yaw-anchor)<.1);
});
test('missing detections stay missing and a long loss does not blend unrelated poses',()=>{
 const times=[0,.04,.08,.12,.16,.20,.24,1,1.04,1.08,1.12,1.16,1.20],raw=times.map(time=>time>.2&&time<1.04?null:pose(time<1?0:35));
 const filtered=smoothYawTrack(track(times),raw,300,0);
 assert.equal(filtered[6],null);assert.equal(filtered[7],null);assert.ok(Math.abs(filtered[5]!.yaw)<1e-8);
});
test('angle wrap is unwrapped before filtering instead of averaging across zero',()=>{
 const times=Array.from({length:49},(_,i)=>i/24),raw=times.map((time,n)=>pose(n%2?179:-179,n)),filtered=smoothYawTrack(track(times),raw,300,180);
 assert.ok(filtered.every(p=>Math.abs(p!.yaw)>175));
});


test('pitch smoothing removes fit noise while preserving yaw and roll exactly',()=>{
 const times=Array.from({length:241},(_,i)=>i/24),t=track(times),clean=times.map(time=>6*Math.sin(time*2*Math.PI/10));
 const raw=times.map((time,n)=>({...pose(Math.sin(time),n),pitch:clean[n]+(n%2?.9:-.9),rotation:yawRotation(clean[n]+(n%2?.9:-.9),Math.sin(time),-2+n*.005)}));
 const filtered=smoothPitchTrack(t,raw,300,0),indices=times.map((_,i)=>i).filter(i=>times[i]>.6&&times[i]<9.4);
 assert.ok(rms(indices.map(i=>filtered[i]!.pitch-clean[i]))<.15);
 for(let n=0;n<raw.length;n++){assert.equal(filtered[n]!.yaw,raw[n].yaw);assert.equal(filtered[n]!.roll,raw[n].roll);assert.equal(filtered[n]!.x,raw[n].x);assert.equal(filtered[n]!.scale,raw[n].scale)}
});

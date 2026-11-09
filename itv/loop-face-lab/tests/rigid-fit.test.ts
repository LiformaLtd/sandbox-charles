import {test} from 'node:test';import assert from 'node:assert/strict';
import {FacePlacement,RIGID_ANCHORS,multiply,projectPoint,rotation} from '../src/lib/pose.ts';
import {smoothSceneRotation} from '../src/lib/rotation-smoothing.ts';
import {fitFrame,fitScene} from '../src/lib/rigid-fit.ts';
import type {FaceFit,FaceFrame,FaceTrack} from '../src/lib/media-types.ts';
const mul=(a:number[],b:number[])=>Array.from({length:9},(_,i)=>{const row=Math.floor(i/3),col=i%3;return a[row*3]*b[col]+a[row*3+1]*b[3+col]+a[row*3+2]*b[6+col]});
function turn(x:number,y:number,z:number){const cx=Math.cos(x),sx=Math.sin(x),cy=Math.cos(y),sy=Math.sin(y),cz=Math.cos(z),sz=Math.sin(z);return mul([cz,-sz,0,sz,cz,0,0,0,1],mul([cy,0,sy,0,1,0,-sy,0,cy],[1,0,0,0,cx,-sx,0,sx,cx]))}
function matrix(r:number[]){const m=new Float32Array(16),s=[1,-1,-1];for(let i=0;i<3;i++)for(let j=0;j<3;j++)m[j*4+i]=r[i*3+j]*s[i]*s[j];m[15]=1;return m}
function source():FaceFit{
 const points=new Float32Array(478*3);for(let i=0;i<478;i++)points.set([.5+Math.cos(i)*.14,.38+Math.sin(i)*.2,Math.sin(i)*.03],i*3);
 const values=[[.39,.3,.01],[.455,.3,-.025],[.545,.3,-.025],[.61,.3,.01],[.5,.315,-.07],[.5,.35,-.1],[.5,.4,-.17],[.5,.38,-.15],[.28,.35,.07],[.72,.35,.07]];
 RIGID_ANCHORS.forEach((i,n)=>points.set(values[n],i*3));points.set([.25,.4,0],234*3);points.set([.75,.4,0],454*3);
 return {points,matrix:matrix(turn(0,0,0))};
}
function frame(p:FacePlacement,r:number[],scale:number,time=0,width=512,height=512,prior=r):FaceFrame{
 const points=new Float32Array(478*3);for(let i=0;i<478;i++){const q=multiply(r,p.canonical.subarray(i*3,i*3+3));points.set([(q[0]*scale+width*.52)/width,(q[1]*scale+height*.4)/height,q[2]*scale/width],i*3)}return {points,matrix:matrix(prior),time};
}
function track(frames:(FaceFrame|null)[],width=512,height=512):FaceTrack{return {frames,width,height,times:frames.map((_,i)=>i/24),duration:frames.length/24,revision:1,missing:frames.filter(v=>!v).length}}

test('one camera scale removes injected size changes while keeping a fixed 3D anchor configuration under rotation',()=>{
 const p=new FacePlacement(source(),512,512),frames=Array.from({length:21},(_,i)=>frame(p,turn(.1*Math.cos(i),.25*Math.sin(i),.08*Math.sin(i*.6)),150*(1+.05*Math.sin(i*Math.PI/10)),i/24)),t=track(frames),original=frames[3].points.slice(),fit=fitScene(p,[t]);
 assert.ok(fit.stats.rawScaleDrift>3);p.attachFit(t,fit.tracks[0].poses);
 const scales=fit.tracks[0].poses.map(v=>v!.scale);assert.ok(Math.max(...scales)-Math.min(...scales)<1e-10);
 assert.ok(Math.abs(scales[0]-150)<.001);
 const pairs=[[33,263],[1,127],[127,356]];
 for(const f of frames){const pose=p.transform(f,512,512),r=pose.rotation;
  for(let i=0;i<3;i++){assert.ok(Math.abs(Math.hypot(r[i*3],r[i*3+1],r[i*3+2])-1)<1e-6);for(let j=i+1;j<3;j++)assert.ok(Math.abs(r[i*3]*r[j*3]+r[i*3+1]*r[j*3+1]+r[i*3+2]*r[j*3+2])<1e-6)}
  for(const [a,b] of pairs){const pa=projectPoint(p.canonical.subarray(a*3,a*3+3),pose),pb=projectPoint(p.canonical.subarray(b*3,b*3+3),pose);const expected=pose.scale*Math.hypot(...[0,1,2].map(k=>p.canonical[a*3+k]-p.canonical[b*3+k]));assert.ok(Math.abs(Math.hypot(...pa.map((v,k)=>v-pb[k]))-expected)<1e-4)}
 }
 assert.deepEqual(frames[3].points,original);
});
test('rotation is actually refitted to image points instead of copying a biased MediaPipe matrix',()=>{
 const p=new FacePlacement(source(),512,512),f=frame(p,turn(.1,.23,-.08),150,0,512,512,turn(.17,.31,-.03)),pose=fitFrame(p,f,512,512,150),prior=rotation(f.matrix);
 assert.ok(pose.fitError!<.35,`RMS ${pose.fitError}`);assert.ok(Math.hypot(...pose.rotation.map((v,i)=>v-prior[i]))>.05);
});
test('a bad nose detection is downweighted; mouth and jaw movement cannot drive the rigid pose',()=>{
 const p=new FacePlacement(source(),512,512),f=frame(p,turn(.1,-.2,.03),150),good=fitFrame(p,f,512,512,150);
 for(const i of [13,14,152,377,148,17])f.points[i*3+1]+=.1;
 assert.deepEqual(fitFrame(p,f,512,512,150),good);
 f.points[1*3]+=.1;const robust=fitFrame(p,f,512,512,150);
 const errors=RIGID_ANCHORS.filter(i=>i!==1).map(i=>{const out=projectPoint(p.canonical.subarray(i*3,i*3+3),robust);return Math.hypot(out[0]-f.points[i*3]*512,out[1]-f.points[i*3+1]*512)});
 assert.ok(errors.reduce((a,b)=>a+b,0)/errors.length<1);assert.ok(robust.fitError!>10);
});
test('speech leaves rigid anchors fixed but still animates the mouth',()=>{
 const p=new FacePlacement(source(),512,512),f=frame(p,turn(.1,.15,.03),150),t=track([f]),fit=fitScene(p,[t]);p.attachFit(t,fit.tracks[0].poses);
 const local=new Float32Array(956);for(let i=0;i<478;i++)local.set([p.canonical[i*3]+.03,p.canonical[i*3+1]+.04],i*2);
 const rest=p.project(f,512,512),spoken=p.project(f,512,512,local);
 for(const i of RIGID_ANCHORS)assert.deepEqual(spoken.points.subarray(i*3,i*3+3),rest.points.subarray(i*3,i*3+3));
 assert.ok(Math.hypot(spoken.points[13*3]-rest.points[13*3],spoken.points[13*3+1]-rest.points[13*3+1])>5);
 p.fixedSize=false;assert.notDeepEqual(p.project(f,512,512,local).points.subarray(33*3,33*3+3),rest.points.subarray(33*3,33*3+3));
});
test('scale is shared across selected clips and missing detections remain missing',()=>{
 const p=new FacePlacement(source(),512,512),a=track([frame(p,turn(.1,0,0),150),null]),b=track([frame(p,turn(0,.2,0),300,0,1024,1024)],1024,1024),fit=fitScene(p,[a,b]);
 assert.equal(fit.tracks[0].poses[1],null);assert.ok(Math.abs(fit.tracks[0].poses[0]!.scale/512-fit.tracks[1].poses[0]!.scale/1024)<1e-12);
});


test('combined yaw and pitch filters keep roll, translation and the fixed camera unchanged',()=>{
 const p=new FacePlacement(source(),512,512),frames=Array.from({length:241},(_,n)=>{
  const time=n/24,noise=n%2?.025:-.025;
  return frame(p,turn(.1*Math.sin(time*Math.PI/5)+noise,.15*Math.sin(time*Math.PI/5)-noise,.03*Math.sin(time)),150,time);
 }),t=track(frames),original=fitScene(p,[t]),filtered=smoothSceneRotation(p,original,{yaw:300,pitch:300});
 assert.ok(filtered.stats.yaw!.reduction>70);assert.ok(filtered.stats.pitch!.reduction>70);
 const a=filtered.fit.tracks[0].poses,b=original.tracks[0].poses;
 for(let n=0;n<a.length;n++)for(const key of ['roll','x','y','scale','centerX','centerY'] as const)assert.equal(a[n]![key],b[n]![key]);
 assert.equal(a[0]!.yaw,filtered.stats.yaw!.endpointAngle);assert.equal(a.at(-1)!.yaw,filtered.stats.yaw!.endpointAngle);
 assert.equal(a[0]!.pitch,filtered.stats.pitch!.endpointAngle);assert.equal(a.at(-1)!.pitch,filtered.stats.pitch!.endpointAngle);
});

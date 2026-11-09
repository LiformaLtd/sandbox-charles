import type {PoseTransform,FacePlacement} from './pose.ts';
import type {FaceTrack} from './media-types';
import {fitErrorForPose} from './rigid-fit.ts';
import type {SceneRigidFit} from './rigid-fit.ts';

export type RotationAxis='yaw'|'pitch';
export type AxisSmoothingStats={before:number;after:number;reduction:number;endpointAngle:number;windowMs:number};
export type RotationSmoothingOptions={yaw?:number;pitch?:number};
export type YawSmoothingStats=AxisSmoothingStats&{endpointYaw:number};
const median=(values:number[])=>{const s=values.slice().sort((a,b)=>a-b),n=s.length;return n%2?s[n>>1]:(s[n/2-1]+s[n/2])/2};
const near=(angle:number,reference:number)=>reference+((angle-reference+180)%360+360)%360-180;
const fade=(value:number)=>{const t=Math.max(0,Math.min(1,value));return t*t*t*(t*(t*6-15)+10)};
export function yawRotation(pitch:number,yaw:number,roll:number){
 const k=Math.PI/180,x=pitch*k,y=yaw*k,z=roll*k,cx=Math.cos(x),sx=Math.sin(x),cy=Math.cos(y),sy=Math.sin(y),cz=Math.cos(z),sz=Math.sin(z);
 return Float32Array.from([cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx,sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx,-sy,cy*sx,cy*cx]);
}
function bounds(track:FaceTrack){const start=track.times[0]??0,end=Math.min(start+10,track.times.at(-1)??track.duration);return {start,end,taper:Math.min(.35,Math.max(.01,(end-start)*.1))}}
function solve3(matrix:Float64Array,rhs:Float64Array){
 for(let c=0;c<3;c++){
  let pivot=c;for(let r=c+1;r<3;r++)if(Math.abs(matrix[r*3+c])>Math.abs(matrix[pivot*3+c]))pivot=r;
  if(Math.abs(matrix[pivot*3+c])<1e-12)throw new Error('Rotation smoothing has insufficient time samples.');
  if(pivot!==c){for(let k=c;k<3;k++)[matrix[c*3+k],matrix[pivot*3+k]]=[matrix[pivot*3+k],matrix[c*3+k]];[rhs[c],rhs[pivot]]=[rhs[pivot],rhs[c]]}
  const d=matrix[c*3+c];for(let k=c;k<3;k++)matrix[c*3+k]/=d;rhs[c]/=d;
  for(let r=0;r<3;r++)if(r!==c){const factor=matrix[r*3+c];for(let k=c;k<3;k++)matrix[r*3+k]-=factor*matrix[c*3+k];rhs[r]-=factor*rhs[c]}
 }return Array.from(rhs);
}
/** Symmetric local quadratic regression: no causal delay and no averaging of matrices. */
export function smoothAngleTrack(track:FaceTrack,poses:(PoseTransform|null)[],windowMs:number,endpointAngle:number,axis:RotationAxis){
 if(poses.length!==track.times.length)throw new Error('Rotation poses do not match video timestamps.');
 if(!Number.isFinite(windowMs)||windowMs<=0)throw new Error('Choose a positive rotation smoothing timescale.');
 const sigma=windowMs/2000,{start,end,taper}=bounds(track),values:number[]=[],segments:number[]=[];
 let previous=endpointAngle,lastTime=-Infinity,segment=0;
 for(let n=0;n<poses.length;n++){
  const pose=poses[n];
  if(pose){
   if(!Number.isFinite(pose[axis]))throw new Error('The fitted rotation track contains an invalid angle.');
   if(track.times[n]-lastTime>.2||(lastTime<=end&&track.times[n]>end))segment++;
   values[n]=near(pose[axis],previous);previous=values[n];lastTime=track.times[n];
  }else values[n]=NaN;
  segments[n]=segment;
 }
 return poses.map((pose,n)=>{
  if(!pose)return null;
  const samples:{x:number;y:number;weight:number}[]=[];
  for(let j=n;j>=0&&track.times[n]-track.times[j]<=sigma*3;j--)if(poses[j]&&segments[j]===segments[n]){
   const x=(track.times[j]-track.times[n])/sigma;samples.push({x,y:values[j]-values[n],weight:Math.exp(-x*x/2)});
  }
  for(let j=n+1;j<poses.length&&track.times[j]-track.times[n]<=sigma*3;j++)if(poses[j]&&segments[j]===segments[n]){
   const x=(track.times[j]-track.times[n])/sigma;samples.push({x,y:values[j]-values[n],weight:Math.exp(-x*x/2)});
  }
  let coefficients=[median(samples.map(s=>s.y)),0,0];
  for(let iteration=0;iteration<4;iteration++){
   const residuals=samples.map(s=>s.y-coefficients[0]-coefficients[1]*s.x-coefficients[2]*s.x*s.x),center=median(residuals);
   const cutoff=Math.max(.35,2.5*1.4826*median(residuals.map(v=>Math.abs(v-center))));
   const matrix=new Float64Array(9),rhs=new Float64Array(3);
   samples.forEach((s,i)=>{
    const weight=s.weight*(iteration===0?1:Math.min(1,cutoff/Math.max(1e-9,Math.abs(residuals[i])))),basis=[1,s.x,s.x*s.x];
    for(let a=0;a<3;a++){rhs[a]+=weight*basis[a]*s.y;for(let b=0;b<3;b++)matrix[a*3+b]+=weight*basis[a]*basis[b]}
   });
   for(let i=0;i<3;i++)matrix[i*3+i]+=1e-8;
   coefficients=solve3(matrix,rhs);
  }
  let angle=values[n]+coefficients[0];
  if(track.times[n]<=end){
   // Same-image endpoints share the filtered angle and ease to zero angular velocity at joins.
   const amount=fade((track.times[n]-start)/taper)*fade((end-track.times[n])/taper),anchor=near(endpointAngle,angle);
   angle=amount===0?endpointAngle:anchor+(angle-anchor)*amount;
  }
  const pitch=axis==='pitch'?angle:pose.pitch,yaw=axis==='yaw'?angle:pose.yaw;
  return {...pose,pitch,yaw,rotation:yawRotation(pitch,yaw,pose.roll)};
 });
}
function acceleration(track:FaceTrack,poses:(PoseTransform|null)[],axis:RotationAxis){
 const {start,end,taper}=bounds(track);let squared=0,count=0;
 for(let n=1;n<poses.length-1;n++){
  const a=poses[n-1],b=poses[n],c=poses[n+1],time=track.times[n];if(!a||!b||!c||time<start+taper||time>end-taper)continue;
  const left=time-track.times[n-1],right=track.times[n+1]-time;if(left<=0||right<=0||left>.2||right>.2)continue;
  const velocityLeft=(near(b[axis],a[axis])-a[axis])/left,velocityRight=(near(c[axis],b[axis])-b[axis])/right;
  squared+=((velocityRight-velocityLeft)/((left+right)/2))**2;count++;
 }return {squared,count};
}
export function smoothYawTrack(track:FaceTrack,poses:(PoseTransform|null)[],windowMs:number,endpointYaw:number){return smoothAngleTrack(track,poses,windowMs,endpointYaw,'yaw')}
export function smoothPitchTrack(track:FaceTrack,poses:(PoseTransform|null)[],windowMs:number,endpointPitch:number){return smoothAngleTrack(track,poses,windowMs,endpointPitch,'pitch')}

/** Smooth selected Euler channels independently, always retaining the original roll. */
export function smoothSceneRotation(placement:FacePlacement,scene:SceneRigidFit,options:RotationSmoothingOptions):{fit:SceneRigidFit;stats:Partial<Record<RotationAxis,AxisSmoothingStats>>}{
 const stats:Partial<Record<RotationAxis,AxisSmoothingStats>>={};let tracks=scene.tracks;
 for(const axis of ['yaw','pitch'] as const){
  const windowMs=options[axis];if(windowMs===undefined)continue;
  const all=scene.tracks.flatMap(entry=>entry.poses.filter(p=>p!==null).map(p=>p[axis]));if(!all.length)throw new Error(`No fitted ${axis} poses to smooth.`);
  const reference=all[0],endpoints:number[]=[];
  for(const entry of scene.tracks){const {start,end}=bounds(entry.track);entry.poses.forEach((p,n)=>{const t=entry.track.times[n];if(p&&t<=end&&(t-start<=.125||end-t<=.125))endpoints.push(near(p[axis],reference))})}
  const endpointAngle=median(endpoints.length?endpoints:all.map(v=>near(v,reference)));
  let beforeSquared=0,afterSquared=0,beforeCount=0,afterCount=0;
  tracks=tracks.map((entry,k)=>{
   const poses=smoothAngleTrack(entry.track,entry.poses,windowMs,endpointAngle,axis);
   const a=acceleration(entry.track,scene.tracks[k].poses,axis),b=acceleration(entry.track,poses,axis);beforeSquared+=a.squared;beforeCount+=a.count;afterSquared+=b.squared;afterCount+=b.count;
   return {track:entry.track,poses};
  });
  const before=Math.sqrt(beforeSquared/Math.max(1,beforeCount)),after=Math.sqrt(afterSquared/Math.max(1,afterCount));
  stats[axis]={before,after,reduction:before>1e-9?(1-after/before)*100:0,endpointAngle,windowMs};
 }
 const errors:number[]=[];
 // With no filters selected, preserve the exact original fit for comparison.
 if(options.yaw===undefined&&options.pitch===undefined)return {fit:scene,stats};
 for(const entry of tracks)entry.poses.forEach((pose,n)=>{if(pose){pose.fitError=fitErrorForPose(placement,entry.track.frames[n]!,entry.track.width,entry.track.height,pose);errors.push(pose.fitError)}});
 return {fit:{tracks,stats:{...scene.stats,medianError:median(errors),maxError:errors.reduce((max,e)=>Math.max(max,e),0)}},stats};
}
export function smoothSceneYaw(placement:FacePlacement,scene:SceneRigidFit,windowMs:number):{fit:SceneRigidFit;stats:YawSmoothingStats}{
 const result=smoothSceneRotation(placement,scene,{yaw:windowMs}),stats=result.stats.yaw!;
 return {fit:result.fit,stats:{...stats,endpointYaw:stats.endpointAngle}};
}

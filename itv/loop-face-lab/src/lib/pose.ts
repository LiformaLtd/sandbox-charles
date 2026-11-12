import type {FaceFit,FaceFrame,FaceTrack} from './media-types';
const STABLE=[33,263,168,6,10,127,356,234,454];
const OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
export {OVAL};
// Eye corners, nose ridge/tip, and temple points. No eyelids, mouth or jaw outline.
export const RIGID_ANCHORS=[33,133,362,263,168,6,1,4,127,356];
const RIGID_SET=new Set(RIGID_ANCHORS);
export function rotation(matrix:ArrayLike<number>){
 const out=new Float32Array(9),sign=[1,-1,-1];
 for(let c=0;c<3;c++){const norm=Math.hypot(matrix[c*4],matrix[c*4+1],matrix[c*4+2]);
  for(let r=0;r<3;r++)out[r*3+c]=matrix[c*4+r]/norm*sign[r]*sign[c];
 }
 // MediaPipe's matrix may contain scale/shear. Keep a proper orthonormal basis.
 const x=[out[0],out[3],out[6]],y=[out[1],out[4],out[7]];
 const dot=x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
 for(let i=0;i<3;i++)y[i]-=dot*x[i];
 const length=Math.hypot(...y);if(length<1e-8)throw new Error('Invalid face rotation matrix.');
 for(let i=0;i<3;i++)y[i]/=length;
 const z=[x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]];
 return Float32Array.from([x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2]]);
}
export function multiply(r:ArrayLike<number>,p:ArrayLike<number>){return [r[0]*p[0]+r[1]*p[1]+r[2]*p[2],r[3]*p[0]+r[4]*p[1]+r[5]*p[2],r[6]*p[0]+r[7]*p[1]+r[8]*p[2]]}
export function transpose(r:ArrayLike<number>){return [r[0],r[3],r[6],r[1],r[4],r[7],r[2],r[5],r[8]]}
export function angles(matrix:ArrayLike<number>){return rotationAngles(rotation(matrix))}
export function rotationAngles(r:ArrayLike<number>){const d=180/Math.PI;return {pitch:Math.atan2(r[7],r[8])*d,yaw:Math.asin(Math.max(-1,Math.min(1,-r[6])))*d,roll:Math.atan2(r[3],r[0])*d}}
export function frameIndexAt(track:FaceTrack,time:number){
 let lo=0,hi=track.times.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(track.times[mid]<=time)lo=mid;else hi=mid-1}return lo;
}
export function frameAt(track:FaceTrack,time:number):FaceFrame|null{return track.frames[frameIndexAt(track,time)]??null}
export type PoseTransform={rotation:Float32Array;scale:number;x:number;y:number;centerX:number;centerY:number;pitch:number;yaw:number;roll:number;fitError?:number};
export function projectPoint(point:ArrayLike<number>,transform:PoseTransform){
 const p=multiply(transform.rotation,point);return [(p[0]-transform.centerX)*transform.scale+transform.x,(p[1]-transform.centerY)*transform.scale+transform.y,p[2]*transform.scale];
}
export class FacePlacement{
 readonly canonical:Float32Array;readonly uv:Float32Array;readonly ring:Float32Array;
 readonly width:number;readonly height:number;
 readonly source:FaceFit;readonly sourceWidth:number;readonly sourceCenter:number[];
 private sourceInverse:number[];
 fixedSize=true;
 private fitted=new WeakMap<FaceFrame,PoseTransform>();
 attachFit(track:FaceTrack,poses:(PoseTransform|null)[]){
  if(poses.length!==track.frames.length)throw new Error('Fixed-size pose count does not match the video.');
  for(let n=0;n<poses.length;n++){const frame=track.frames[n],pose=poses[n];if(frame&&pose)this.fitted.set(frame,pose)}
 }
 transform(frame:FaceFrame,width:number,height:number):PoseTransform{
  const fitted=this.fixedSize?this.fitted.get(frame):undefined;
  return fitted??this.rawTransform(frame,width,height);
 }
 constructor(source:FaceFit,width:number,height:number){
  this.width=width;this.height=height;
  this.source=source;this.sourceInverse=transpose(rotation(source.matrix));const center=[0,0,0];
  for(const i of STABLE){center[0]+=source.points[i*3]*width/STABLE.length;center[1]+=source.points[i*3+1]*height/STABLE.length;center[2]+=source.points[i*3+2]*width/STABLE.length}
  const size=Math.hypot((source.points[454*3]-source.points[234*3])*width,(source.points[454*3+1]-source.points[234*3+1])*height);
  if(size<8)throw new Error('The source face is too small. Choose a larger face image.');
  this.sourceWidth=size;this.sourceCenter=center;
  this.canonical=new Float32Array(478*3);this.uv=new Float32Array(478*2);
  for(let i=0;i<478;i++){
   this.canonical.set(this.sourcePoint(source.points[i*3],source.points[i*3+1],source.points[i*3+2]),i*3);
   this.uv.set([source.points[i*3],source.points[i*3+1]],i*2);
  }
  this.ring=new Float32Array(OVAL.length*3);
  for(let n=0;n<OVAL.length;n++){const i=OVAL[n];this.ring.set([this.canonical[i*3]*1.18,this.canonical[i*3+1]*1.18,this.canonical[i*3+2]],n*3)}
 }
 sourcePoint(u:number,v:number,z:number){
  return multiply(this.sourceInverse,[(u*this.width-this.sourceCenter[0])/this.sourceWidth,(v*this.height-this.sourceCenter[1])/this.sourceWidth,(z*this.width-this.sourceCenter[2])/this.sourceWidth]);
 }
 rawTransform(frame:FaceFrame,width:number,height:number):PoseTransform{
  const r=rotation(frame.matrix),base=this.canonical,predicted=[0,0],observed=[0,0];
  for(const i of STABLE){const p=multiply(r,base.subarray(i*3,i*3+3));predicted[0]+=p[0]/STABLE.length;predicted[1]+=p[1]/STABLE.length;observed[0]+=frame.points[i*3]*width/STABLE.length;observed[1]+=frame.points[i*3+1]*height/STABLE.length}
  let numerator=0,denominator=0;
  for(const i of STABLE){const p=multiply(r,base.subarray(i*3,i*3+3));const x=p[0]-predicted[0],y=p[1]-predicted[1];numerator+=x*(frame.points[i*3]*width-observed[0])+y*(frame.points[i*3+1]*height-observed[1]);denominator+=x*x+y*y}
  return {rotation:r,scale:numerator/Math.max(1e-8,denominator),x:observed[0],y:observed[1],centerX:predicted[0],centerY:predicted[1],...angles(frame.matrix)};
 }
 project(frame:FaceFrame,width:number,height:number,local?:Float32Array){
  const transform=this.transform(frame,width,height),points=new Float32Array((478+36)*3),base=this.canonical;
  for(let i=0;i<478+36;i++){
   const pinned=this.fixedSize&&this.fitted.has(frame)&&RIGID_SET.has(i);
   const p=i<478?[pinned?base[i*3]:local?.[i*2]??base[i*3],pinned?base[i*3+1]:local?.[i*2+1]??base[i*3+1],base[i*3+2]]:this.ring.subarray((i-478)*3,(i-478)*3+3);
   points.set(projectPoint(p,transform),i*3);
  }
  return {points,...transform};
 }
}

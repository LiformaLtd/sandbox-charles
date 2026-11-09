import {RIGID_ANCHORS,multiply,rotation,rotationAngles,projectPoint} from './pose.ts';
import type {FacePlacement,PoseTransform} from './pose.ts';
import type {FaceFrame,FaceTrack} from './media-types';

const WEIGHTS=[1,1,1,1,1,.8,.8,.8,.6,.6];
const PRIOR=.012;
export type RigidFitStats={scale:number;frames:number;rawScaleDrift:number;medianError:number;maxError:number};
export type SceneRigidFit={stats:RigidFitStats;tracks:{track:FaceTrack;poses:(PoseTransform|null)[]}[]};
const median=(values:number[])=>{const sorted=values.slice().sort((a,b)=>a-b),n=sorted.length;return n%2?sorted[n>>1]:(sorted[n/2-1]+sorted[n/2])/2};
const product=(a:ArrayLike<number>,b:ArrayLike<number>)=>Array.from({length:9},(_,i)=>{const row=Math.floor(i/3),col=i%3;return a[row*3]*b[col]+a[row*3+1]*b[3+col]+a[row*3+2]*b[6+col]});
const transpose=(r:ArrayLike<number>)=>[r[0],r[3],r[6],r[1],r[4],r[7],r[2],r[5],r[8]];
function exp(v:number[]){
 const angle=Math.hypot(...v);if(angle<1e-12)return [1,0,0,0,1,0,0,0,1];
 const [x,y,z]=v.map(value=>value/angle),c=Math.cos(angle),s=Math.sin(angle),d=1-c;
 return [c+x*x*d,x*y*d-z*s,x*z*d+y*s,y*x*d+z*s,c+y*y*d,y*z*d-x*s,z*x*d-y*s,z*y*d+x*s,c+z*z*d];
}
function log(r:ArrayLike<number>){
 const angle=Math.acos(Math.max(-1,Math.min(1,(r[0]+r[4]+r[8]-1)/2))),factor=angle<1e-7?.5:angle/(2*Math.sin(angle));
 return [(r[7]-r[5])*factor,(r[2]-r[6])*factor,(r[3]-r[1])*factor];
}
function solve(matrix:Float64Array,rhs:Float64Array){
 const n=rhs.length;
 for(let c=0;c<n;c++){
  let pivot=c;for(let r=c+1;r<n;r++)if(Math.abs(matrix[r*n+c])>Math.abs(matrix[pivot*n+c]))pivot=r;
  if(Math.abs(matrix[pivot*n+c])<1e-12)throw new Error('Rigid face fit is underconstrained.');
  if(pivot!==c){for(let k=c;k<n;k++)[matrix[c*n+k],matrix[pivot*n+k]]=[matrix[pivot*n+k],matrix[c*n+k]];[rhs[c],rhs[pivot]]=[rhs[pivot],rhs[c]]}
  const d=matrix[c*n+c];for(let k=c;k<n;k++)matrix[c*n+k]/=d;rhs[c]/=d;
  for(let r=0;r<n;r++)if(r!==c){const factor=matrix[r*n+c];for(let k=c;k<n;k++)matrix[r*n+k]-=factor*matrix[c*n+k];rhs[r]-=factor*rhs[c]}
 }
 return Array.from(rhs);
}
function observations(frame:FaceFrame,width:number,height:number){
 return RIGID_ANCHORS.map(i=>{
  const point=[frame.points[i*3]*width,frame.points[i*3+1]*height];
  if(!point.every(Number.isFinite))throw new Error(`Invalid eye/nose/temple tracking at ${frame.time.toFixed(3)}s.`);
  return point;
 });
}
function reference(placement:FacePlacement){return RIGID_ANCHORS.map(i=>Array.from(placement.canonical.subarray(i*3,i*3+3)))}
function error(r:ArrayLike<number>,t:number[],model:number[][],observed:number[][]){
 return model.map((p,i)=>{const q=multiply(r,p);return [observed[i][0]-q[0]-t[0],observed[i][1]-q[1]-t[1]]});
}
const huber=(norm:number,delta:number)=>norm<=delta?norm*norm:2*delta*norm-delta*delta;
/** Robust size measurement is only used once to calibrate the shared camera. */
export function measureScale(model:number[][],observed:number[][],r:ArrayLike<number>){
 const projected=model.map(p=>multiply(r,p));let weights=WEIGHTS.slice(),scale=1,t=[0,0];
 for(let iteration=0;iteration<5;iteration++){
  let total=0,px=0,py=0,ox=0,oy=0;
  weights.forEach((w,i)=>{total+=w;px+=projected[i][0]*w;py+=projected[i][1]*w;ox+=observed[i][0]*w;oy+=observed[i][1]*w});
  px/=total;py/=total;ox/=total;oy/=total;let numerator=0,denominator=0;
  weights.forEach((w,i)=>{const x=projected[i][0]-px,y=projected[i][1]-py;numerator+=w*(x*(observed[i][0]-ox)+y*(observed[i][1]-oy));denominator+=w*(x*x+y*y)});
  scale=numerator/Math.max(1e-12,denominator);t=[ox-scale*px,oy-scale*py];
  if(!(scale>0)||!Number.isFinite(scale))throw new Error('Cannot calibrate a positive fixed face scale.');
  weights=WEIGHTS.map((w,i)=>w*Math.min(1,Math.max(1.5,scale*.015)/Math.max(1e-9,Math.hypot(observed[i][0]-projected[i][0]*scale-t[0],observed[i][1]-projected[i][1]*scale-t[1]))));
 }
 return scale;
}
/** Orthographic five-DOF fit: proper 3D rotation + 2D translation, fixed scale. */
export function fitFrame(placement:FacePlacement,frame:FaceFrame,width:number,height:number,scale:number):PoseTransform{
 const model=reference(placement),pixels=observations(frame,width,height),observed=pixels.map(p=>p.map(v=>v/scale)),prior=rotation(frame.matrix);
 let r=Array.from(prior),t=[0,0];const total=WEIGHTS.reduce((a,b)=>a+b,0);
 model.forEach((p,i)=>{const q=multiply(r,p);t[0]+=(observed[i][0]-q[0])*WEIGHTS[i]/total;t[1]+=(observed[i][1]-q[1])*WEIGHTS[i]/total});
 const delta=Math.max(1.5/scale,.015);
 const objective=(candidate:number[],translation:number[])=>{
  const residual=error(candidate,translation,model,observed),angle=log(product(prior,transpose(candidate)));
  return residual.reduce((sum,e,i)=>sum+WEIGHTS[i]*huber(Math.hypot(...e),delta),0)+PRIOR*angle.reduce((sum,v)=>sum+v*v,0);
 };
 for(let iteration=0;iteration<20;iteration++){
  const matrix=new Float64Array(25),rhs=new Float64Array(5),residual=error(r,t,model,observed);
  for(let i=0;i<model.length;i++){
   const q=multiply(r,model[i]),w=WEIGHTS[i]*Math.min(1,delta/Math.max(1e-10,Math.hypot(...residual[i]))),jacobian=[[0,q[2],-q[1],1,0],[-q[2],0,q[0],0,1]];
   for(let a=0;a<5;a++)for(let axis=0;axis<2;axis++){
    rhs[a]+=w*jacobian[axis][a]*residual[i][axis];for(let b=0;b<5;b++)matrix[a*5+b]+=w*jacobian[axis][a]*jacobian[axis][b];
   }
  }
  const toward=log(product(prior,transpose(r)));
  for(let i=0;i<5;i++)matrix[i*5+i]+=1e-6;
  for(let i=0;i<3;i++){matrix[i*5+i]+=PRIOR;rhs[i]+=PRIOR*toward[i]}
  const step=solve(matrix,rhs),angle=Math.hypot(...step.slice(0,3));if(angle>.12)for(let i=0;i<3;i++)step[i]*=.12/angle;
  const previous=objective(r,t);let accepted=false;
  for(let factor=1;factor>=1/32;factor/=2){
   const candidate=product(exp(step.slice(0,3).map(v=>v*factor)),r),translation=[t[0]+step[3]*factor,t[1]+step[4]*factor];
   if(objective(candidate,translation)<=previous+1e-14){r=candidate;t=translation;accepted=true;break}
  }
  if(!accepted||Math.hypot(...step)<1e-7)break;
 }
 const residual=error(r,t,model,observed),rms=scale*Math.sqrt(residual.reduce((sum,e,i)=>sum+WEIGHTS[i]*(e[0]*e[0]+e[1]*e[1]),0)/total);
 return {rotation:Float32Array.from(r),scale,x:t[0]*scale,y:t[1]*scale,centerX:0,centerY:0,...rotationAngles(r),fitError:rms};
}
export function fitErrorForPose(placement:FacePlacement,frame:FaceFrame,width:number,height:number,pose:PoseTransform){
 const observed=observations(frame,width,height);let squared=0,total=0;
 RIGID_ANCHORS.forEach((point,i)=>{
  const p=projectPoint(placement.canonical.subarray(point*3,point*3+3),pose),w=WEIGHTS[i];
  squared+=w*((p[0]-observed[i][0])**2+(p[1]-observed[i][1])**2);total+=w;
 });return Math.sqrt(squared/total);
}

export function fitScene(placement:FacePlacement,tracks:FaceTrack[]):SceneRigidFit{
 const model=reference(placement),measurements:number[]=[];
 for(const track of tracks)for(const frame of track.frames)if(frame)measurements.push(measureScale(model,observations(frame,track.width,track.height),rotation(frame.matrix))/Math.min(track.width,track.height));
 if(!measurements.length)throw new Error('No tracked eyes, nose and temple points are available for fixed-size fitting.');
 const scale=median(measurements),errors:number[]=[],fitted=tracks.map(track=>{
  const poses=track.frames.map(frame=>frame?fitFrame(placement,frame,track.width,track.height,scale*Math.min(track.width,track.height)):null);
  for(const pose of poses)if(pose)errors.push(pose.fitError!);
  return {track,poses};
 });
 const mean=measurements.reduce((sum,v)=>sum+v,0)/measurements.length;
 return {tracks:fitted,stats:{scale,frames:measurements.length,rawScaleDrift:Math.sqrt(measurements.reduce((sum,v)=>sum+(v-mean)**2,0)/measurements.length)/mean*100,medianError:median(errors),maxError:errors.reduce((max,value)=>Math.max(max,value),0)}};
}

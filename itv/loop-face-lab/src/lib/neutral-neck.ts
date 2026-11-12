import {FacePlacement,OVAL,frameIndexAt,multiply,projectPoint,rotation} from './pose.ts';
import type {PoseTransform} from './pose.ts';
import type {FaceTrack} from './media-types';

const ROWS=8;
const clamp=(v:number,lo=0,hi=1)=>Math.max(lo,Math.min(hi,v));
const smooth=(v:number)=>{const t=clamp(v);return t*t*(3-2*t)};
export type NeutralGeometry={uv:Float32Array;canonical:Float32Array;alpha:Float32Array;motion:Float32Array;triangles:number[]};
type PreparedTrack={poses:(PoseTransform|null)[];anchors:PoseTransform[]};

/** A neutral face plus neck skirt. Coverage, motion weights and alpha are independent. */
export class NeutralNeckLayer{
 geometry:NeutralGeometry;
 extent=.5;requiredExtent=.18;limited=false;
 private tracks=new WeakMap<FaceTrack,PreparedTrack>();
 private template:number[];
 private placement:FacePlacement;
 constructor(placement:FacePlacement,triangles:number[]){
  this.placement=placement;
  this.template=[];
  // Replace the old single feather ring with a graded eight-row skirt.
  for(let t=0;t<triangles.length;t+=3)if(triangles.slice(t,t+3).every(i=>i<478))this.template.push(...triangles.slice(t,t+3));
  this.geometry=this.build(.5);
 }
 prepare(track:FaceTrack){
  const placement=this.placement;
  const poses=track.frames.map(frame=>frame?placement.transform(frame,track.width,track.height):null);
  const first=poses.find(p=>p!==null);if(!first)throw new Error('No valid head poses for the neutral neck layer.');
  const anchors:PoseTransform[]=[];
  const sourceRotation=rotation(placement.source.matrix);
  let centerX=0,centerY=0;
  for(const i of [33,263,168,6,10,127,356,234,454]){const p=multiply(sourceRotation,placement.canonical.subarray(i*3,i*3+3));centerX+=p[0]/9;centerY+=p[1]/9}
  for(let n=0;n<poses.length;n++){
   // Symmetric smoothing is available because the complete track is precomputed.
   // This is an image-plane anchor, not an independently measured torso pose.
   let x=0,y=0,scale=0,total=0;
   for(let j=Math.max(0,n-24);j<=Math.min(poses.length-1,n+24);j++){
    const p=poses[j];if(!p)continue;const w=Math.exp(-(((track.times[j]-track.times[n])/.45)**2));
    x+=p.x*w;y+=p.y*w;scale+=p.scale*w;total+=w;
   }
   anchors.push({...first,rotation:sourceRotation,centerX,centerY,
    x:first.x+(x/total-first.x)*.35,y:first.y+(y/total-first.y)*.35,
    scale:first.scale+(scale/total-first.scale)*.35});
   const frame=track.frames[n],pose=poses[n];if(!frame||!pose)continue;
   const down=multiply(pose.rotation,[0,1,0]),length=Math.hypot(down[0],down[1]);
   for(const i of OVAL.slice(9,28)){
    const neutral=projectPoint(placement.canonical.subarray(i*3,i*3+3),pose);
    const drop=((frame.points[i*3]*track.width-neutral[0])*down[0]+(frame.points[i*3+1]*track.height-neutral[1])*down[1])/Math.max(1e-6,pose.scale*length);
    // Keep the original jaw inside opaque rows, before the outer alpha fade.
    this.requiredExtent=Math.max(this.requiredExtent,(drop+.055)/.58);
   }
  }
  this.tracks.set(track,{poses,anchors});
 }
 setExtent(requested:number){this.geometry=this.build(Math.max(requested,this.requiredExtent));return this.geometry}
 private build(requested:number):NeutralGeometry{
  const p=this.placement,count=478+ROWS*36,uv=new Float32Array(count*2),canonical=new Float32Array(count*3),alpha=new Float32Array(count),motion=new Float32Array(count);
  uv.set(p.uv);canonical.set(p.canonical);alpha.fill(1,0,478);motion.fill(1,0,478);
  const available=Math.max(.02,(.985-p.uv[152*2+1])*p.height/p.sourceWidth);
  this.extent=Math.min(requested,available);this.limited=requested>available;
  const triangles=this.template.slice();let cx=0,cy=0;
  for(const i of OVAL){cx+=p.uv[i*2]/36;cy+=p.uv[i*2+1]/36}
  for(let row=1;row<=ROWS;row++)for(let n=0;n<36;n++){
   const i=OVAL[n],v=478+(row-1)*36+n,t=row/ROWS;
   const lower=n>=8&&n<=28?Math.sin(Math.PI*(n-8)/20):0;
   // Reach into source-image neck pixels; keep source UV and geometry in agreement.
   const u=clamp(cx+(p.uv[i*2]-cx)*(1+.18*t));
   const y=clamp(cy+(p.uv[i*2+1]-cy)*(1+.10*t)+this.extent*p.sourceWidth/p.height*t*lower);
   const z=p.source.points[i*3+2]*(1-t*lower)+(p.source.points[152*3+2]+.12*p.sourceWidth/p.width)*t*lower;
   uv.set([u,y],v*2);canonical.set(p.sourcePoint(u,y,z),v*3);
   const neck=smooth(lower/.45);
   motion[v]=1-neck*smooth(t/.9);
   // Motion fades early; opacity stays solid through the old jaw's coverage band.
   alpha[v]=1-smooth((t-(.66*neck))/(1-.66*neck));
   const next=(n+1)%36,a=row===1?i:478+(row-2)*36+n,b=row===1?OVAL[next]:478+(row-2)*36+next,c=478+(row-1)*36+next;
   triangles.push(a,b,v,b,c,v);
  }
  return {uv,canonical,alpha,motion,triangles};
 }
 project(track:FaceTrack,time:number){
  const prepared=this.tracks.get(track);if(!prepared)throw new Error('The neutral neck track was not prepared.');
  const n=frameIndexAt(track,time),pose=prepared.poses[n];if(!pose)return null;
  const anchor=prepared.anchors[n],geometry=this.geometry,points=new Float32Array(geometry.canonical.length);
  for(let i=0;i<geometry.motion.length;i++){
   const source=geometry.canonical.subarray(i*3,i*3+3),head=projectPoint(source,pose),body=projectPoint(source,anchor),weight=geometry.motion[i];
   for(let axis=0;axis<3;axis++)points[i*3+axis]=body[axis]+(head[axis]-body[axis])*weight;
  }
  return points;
 }
}

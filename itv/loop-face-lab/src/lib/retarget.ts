import {parseMapping,ShapeSampler} from './shape';
import type {AvatarMesh,ContactRig,Mapping} from './types';
import {FacePlacement,OVAL} from './pose';

/** Retarget the existing calibrated linear displacement field to a newly fitted face. */
export function retarget(template:Mapping,placement:FacePlacement,triangles:number[]):Mapping{
 const reference=new Float32Array(956),basis=new Float32Array(956*52);
 for(let i=0;i<478;i++)reference.set(placement.canonical.subarray(i*3,i*3+2),i*2);
 for(let i=0;i<478;i++){
  const neighbors=Array.from({length:478},(_,j)=>j).filter(j=>j!==i).sort((a,b)=>Math.hypot(template.reference[a*2]-template.reference[i*2],template.reference[a*2+1]-template.reference[i*2+1])-Math.hypot(template.reference[b*2]-template.reference[i*2],template.reference[b*2+1]-template.reference[i*2+1])).slice(0,16);
  let xx=1e-6,xy=0,yy=1e-6,ux=0,uy=0,vx=0,vy=0;
  for(const j of neighbors){const x=template.reference[j*2]-template.reference[i*2],y=template.reference[j*2+1]-template.reference[i*2+1],u=reference[j*2]-reference[i*2],v=reference[j*2+1]-reference[i*2+1];xx+=x*x;xy+=x*y;yy+=y*y;ux+=u*x;uy+=u*y;vx+=v*x;vy+=v*y}
  const det=xx*yy-xy*xy;
  const a=Math.max(-3,Math.min(3,(ux*yy-uy*xy)/det)),b=Math.max(-3,Math.min(3,(uy*xx-ux*xy)/det));
  const c=Math.max(-3,Math.min(3,(vx*yy-vy*xy)/det)),d=Math.max(-3,Math.min(3,(vy*xx-vx*xy)/det));
  for(let channel=0;channel<52;channel++){const dx=template.basis[i*104+channel],dy=template.basis[i*104+52+channel];basis[i*104+channel]=a*dx+b*dy;basis[i*104+52+channel]=c*dx+d*dy}
 }
 return {...template,kind:'linear',reference,basis};
}
export function createSampler(placement:FacePlacement,templateMesh:AvatarMesh,templateData:Record<string,unknown>,contact:ContactRig){
 const map=retarget(parseMapping(templateData),placement,templateMesh.triangles);
 const mesh:AvatarMesh={...templateMesh,sourceUV:Array.from(placement.uv),cheek:undefined,
  frame:{center:[0,0],width:1,rotation:[1,0,0,1]},ring:OVAL.flatMap((_,n)=>[placement.ring[n*3],placement.ring[n*3+1]])};
 return new ShapeSampler(mesh,map,contact);
}

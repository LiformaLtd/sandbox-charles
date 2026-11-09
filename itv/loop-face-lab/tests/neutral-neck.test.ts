import {test} from 'node:test';
import assert from 'node:assert/strict';
import {FacePlacement,OVAL} from '../src/lib/pose.ts';
import {NeutralNeckLayer} from '../src/lib/neutral-neck.ts';
import type {FaceFit,FaceTrack} from '../src/lib/media-types.ts';
function source():FaceFit{
 const points=new Float32Array(478*3);
 for(let i=0;i<478;i++)points.set([.5+Math.cos(i)*.1,.35+Math.sin(i)*.1,0],i*3);
 for(let n=0;n<36;n++){const a=n*Math.PI/18;points.set([.5+Math.sin(a)*.18,.35-Math.cos(a)*.23,0],OVAL[n]*3)}
 return {points,matrix:Float32Array.from([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])};
}
function track(fit:FaceFit):FaceTrack{return {revision:1,width:512,height:512,duration:.1,times:[0],frames:[{...fit,time:0}],missing:0}}
function inside(x:number,y:number,p:Float32Array,a:number,b:number,c:number){
 const ax=p[a*3],ay=p[a*3+1],bx=p[b*3],by=p[b*3+1],cx=p[c*3],cy=p[c*3+1],d=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy);
 if(Math.abs(d)<1e-8)return null;
 const u=((by-cy)*(x-cx)+(cx-bx)*(y-cy))/d,v=((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/d,w=1-u-v;
 return Math.min(u,v,w)>=-1e-6?[u,v,w]:null;
}
test('the complete neutral skirt samples the matching source pixels without shifting the jaw',()=>{
 const fit=source(),placement=new FacePlacement(fit,512,512),layer=new NeutralNeckLayer(placement,[]),t=track(fit);
 layer.prepare(t);layer.setExtent(.5);const output=layer.project(t,0)!,g=layer.geometry;
 assert.equal(g.motion.length,478+8*36);
 for(let i=0;i<g.motion.length;i++){assert.ok(Math.abs(output[i*3]-g.uv[i*2]*512)<.001);assert.ok(Math.abs(output[i*3+1]-g.uv[i*2+1]*512)<.001)}
 for(let i=0;i<478;i++){assert.equal(g.motion[i],1);assert.equal(g.alpha[i],1)}
});
test('neck motion reduces before opacity fades; the lower boundary has zero head rotation influence',()=>{
 const layer=new NeutralNeckLayer(new FacePlacement(source(),512,512),[]),g=layer.geometry;
 const weights=Array.from({length:8},(_,r)=>g.motion[478+r*36+18]);
 assert.ok(weights.every((w,i)=>i===0||w<=weights[i-1]));assert.equal(weights[7],0);
 const middle=478+4*36+18;assert.ok(g.motion[middle]<.3);assert.equal(g.alpha[middle],1);
 assert.equal(g.alpha[478+7*36+18],0);
});
test('an opening video jaw remains under opaque neutral neck pixels and cannot move the neutral face',()=>{
 const fit=source(),placement=new FacePlacement(fit,512,512),layer=new NeutralNeckLayer(placement,[]),closed=track(fit),open=track({...fit,points:fit.points.slice()});
 open.frames[0]!.points[152*3+1]+=.11;
 layer.prepare(closed);layer.prepare(open);layer.setExtent(.2);
 assert.ok(layer.extent>.2);assert.ok(layer.requiredExtent>.2);
 const output=layer.project(open,0)!,g=layer.geometry,x=open.frames[0]!.points[152*3]*512,y=open.frames[0]!.points[152*3+1]*512;
 let opacity=0;
 for(let n=0;n<g.triangles.length;n+=3){const ids=g.triangles.slice(n,n+3),b=inside(x,y,output,ids[0],ids[1],ids[2]);if(b)opacity=Math.max(opacity,b.reduce((sum,w,i)=>sum+w*g.alpha[ids[i]],0))}
 assert.ok(opacity>.999,`Original lowered chin must be covered opaquely, got ${opacity}`);
 assert.deepEqual(layer.project(open,0),layer.project(closed,0));
});
test('the neutral layer never samples outside a tightly cropped source; limited neck coverage is explicit',()=>{
 const fit=source();for(let i=0;i<478;i++)fit.points[i*3+1]=.58+fit.points[i*3+1]*.7;
 const layer=new NeutralNeckLayer(new FacePlacement(fit,512,512),[]);layer.setExtent(1);
 assert.equal(layer.limited,true);assert.ok(layer.extent<1);assert.ok(layer.geometry.uv.every(v=>v>=0&&v<=1));
});

test('head roll moves the neutral face while leaving the lower neck anchor unrotated',()=>{
 const fit=source(),placement=new FacePlacement(fit,512,512),moved={points:fit.points.slice(),matrix:fit.matrix.slice()},a=.25,c=Math.cos(a),s=Math.sin(a);
 moved.matrix.set([c,-s,0,0,s,c,0,0,0,0,1,0,0,0,0,1]);
 for(let i=0;i<478;i++){
  const x=fit.points[i*3]*512-placement.sourceCenter[0],y=fit.points[i*3+1]*512-placement.sourceCenter[1];
  moved.points[i*3]=(placement.sourceCenter[0]+c*x-s*y)/512;moved.points[i*3+1]=(placement.sourceCenter[1]+s*x+c*y)/512;
 }
 const t=track(fit);t.frames.push({...moved,time:.04});t.times.push(.04);
 const layer=new NeutralNeckLayer(placement,[]);layer.prepare(t);layer.setExtent(.5);
 const before=layer.project(t,0)!,after=layer.project(t,.04)!,bottom=478+7*36+18;
 assert.ok(Math.hypot(after[152*3]-before[152*3],after[152*3+1]-before[152*3+1])>5);
 assert.ok(Math.hypot(after[bottom*3]-before[bottom*3],after[bottom*3+1]-before[bottom*3+1])<.001);
});

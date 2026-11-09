import {test} from 'node:test';import assert from 'node:assert/strict';
import {FacePlacement,frameAt,rotation,angles} from '../src/lib/pose.ts';
import type {FaceFit,FaceTrack} from '../src/lib/media-types.ts';
function fit():FaceFit{
 const points=new Float32Array(478*3);for(let i=0;i<478;i++){points[i*3]=.5+Math.cos(i)*.17;points[i*3+1]=.5+Math.sin(i)*.21;points[i*3+2]=Math.sin(i*.3)*.04}
 points[234*3]=.3;points[454*3]=.7;
 const matrix=Float32Array.from([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);return {points,matrix};
}
test('neutral projection reconstructs the source fit exactly, including non-square images',()=>{
 const source=fit(),placement=new FacePlacement(source,768,512),out=placement.project({...source,time:0},768,512);
 for(let i=0;i<478;i++){assert.ok(Math.abs(out.points[i*3]-source.points[i*3]*768)<.001);assert.ok(Math.abs(out.points[i*3+1]-source.points[i*3+1]*512)<.001)}
});
test('video face translation follows rigid anchors; video mouth expressions cannot drive the replacement',()=>{
 const source=fit(),placement=new FacePlacement(source,512,512),frame={points:source.points.slice(),matrix:source.matrix,time:0};
 for(let i=0;i<478;i++){frame.points[i*3]+=.1;frame.points[i*3+1]+=.05}
 const out=placement.project(frame,512,512);frame.points[13*3+1]-=.05;frame.points[14*3+1]+=.1;
 const changed=placement.project(frame,512,512);assert.deepEqual(changed.points,out.points);
 assert.ok(Math.abs(out.points[0]-source.points[0]*512-51.2)<.001);
});
test('pose extraction removes matrix scale and preserves a known roll rotation',()=>{
 const a=.3,c=Math.cos(a),s=Math.sin(a),matrix=Float32Array.from([2*c,2*s,0,0,-2*s,2*c,0,0,0,0,2,0,0,0,0,1]);
 assert.ok(Math.abs(angles(matrix).roll+a*180/Math.PI)<.001);assert.ok(Math.abs(rotation(matrix)[0]-c)<.00001);
});
test('frame selection uses presentation timestamps and explicitly preserves missing-face frames',()=>{
 const first={...fit(),time:0},last={...fit(),time:.09};const track={times:[0,.04,.09],frames:[first,null,last]} as FaceTrack;
 assert.equal(frameAt(track,.039),first);assert.equal(frameAt(track,.04),null);assert.equal(frameAt(track,100),last);
});

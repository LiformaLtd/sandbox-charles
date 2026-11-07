import {test} from 'node:test';
import assert from 'node:assert/strict';
import channels from '../src/lib/studio/channels.json' with {type:'json'};
import {CHANNELS,FPS} from '../src/lib/dictionary/index.mjs';
import {makeSZShapeRegions,applySZShapeConfidence} from '../src/lib/dictionary/sz-shape-confidence.mjs';
const jaw=channels.indexOf('jawOpen');

function fixture(changed=true) {
  const phones=['AA','K','S','JH','AH'].map((phone,i)=>({phone,start:(i+1)/10,end:(i+2)/10}));
  const weights=new Float32Array(180*CHANNELS);
  const clips=[['AA','K','S'],['AA','K','S'],['Z','JH','AH']].map((labels,i)=>({id:'donor-'+i,frameOffset:i*60,frameCount:60,phones:labels.map((phone,j)=>({phone,start:(j+1)/10,end:(j+2)/10}))}));
  for(let f=0;f<180;f++)weights[f*CHANNELS+jaw]=f<60?.8:.2;
  const matches=clips.map(clip=>({clip,source:clip.phones,count:3}));matches[2].approximate={pair:'S/Z'};
  const segment=(match,at)=>({first:phones[at],pieces:[{match,first:match.source[0]}]});
  const plan={phones,runs:[{from:0,to:5}],segments:[segment(matches[0],0),segment(matches[2],2)]};
  const reference={...plan,segments:[segment(matches[changed?1:0],0)]};
  const dictionary={weights,lipRig:{neutral:[0],deltas:Array.from({length:CHANNELS},(_,i)=>[i===jaw?1:0])}};
  return{dictionary,plan,reference,matches};
}

test('an S/Z choice that changes an exact donor gets a geometry-dependent confidence cap',()=>{
  const {dictionary,plan,reference}=fixture();
  const before=structuredClone(plan.phones),regions=makeSZShapeRegions(dictionary,plan,reference);
  assert.equal(regions.length,1);assert.equal(regions[0].changedExactTriples.length,1);
  assert.ok(regions[0].confidence>0&&regions[0].confidence<.25);
  assert.ok(regions[0].shapeDistance>0);assert.equal(regions[0].start,.1);assert.equal(regions[0].end,.6);
  assert.deepEqual(plan.phones,before);
});

test('S/Z retains its full contribution when exact donors stay unchanged',()=>{
  const {dictionary,plan,reference}=fixture(false);
  const regions=makeSZShapeRegions(dictionary,plan,reference);
  assert.equal(regions[0].confidence,1);
  const frames=new Float32Array(60*CHANNELS).fill(.4),original=frames.slice();
  applySZShapeConfidence(dictionary,frames,new Float32Array(frames.length).fill(.2),regions);
  assert.deepEqual(frames,original);
});

test('fallback pieces cannot overwrite the target position of an exact match',()=>{
  const {dictionary,plan,reference,matches}=fixture();
  plan.segments.push({kind:'fallback',first:plan.phones[3],pieces:[{match:matches[0],first:matches[0].source[2]}]});
  assert.equal(makeSZShapeRegions(dictionary,plan,reference)[0].changedExactTriples[0].at,0);
});

test('confidence blending stays bounded and eases back to the current track at the edges',()=>{
  const {dictionary}=fixture();const regions=[{start:.3,end:.6,confidence:.2}];
  const frames=new Float32Array(60*CHANNELS).fill(.5),reference=new Float32Array(frames.length).fill(.2),before=frames.slice();
  const d=applySZShapeConfidence(dictionary,frames,reference,regions);
  assert.ok(d.changedFrames>0);assert.ok(Math.abs(frames[24*CHANNELS+jaw]-.26)<1e-6);
  assert.equal(frames[15*CHANNELS+jaw],before[15*CHANNELS+jaw]);
  assert.equal(frames[40*CHANNELS+jaw],before[40*CHANNELS+jaw]);
  assert.ok(frames[17*CHANNELS+jaw]>frames[18*CHANNELS+jaw]);
  assert.ok(frames.every(v=>Number.isFinite(v)&&v>=0&&v<=1));
});

test('confidence restoration cannot lose or add a contact frame, including overlapping regions',()=>{
  const {dictionary}=fixture();const regions=[{start:0,end:1,confidence:.1},{start:.2,end:.8,confidence:.2}];
  const frames=new Float32Array(60*CHANNELS),reference=new Float32Array(frames.length);
  for(let f=0;f<60;f++){frames[f*CHANNELS+jaw]=f%2?.5:0;reference[f*CHANNELS+jaw]=f%2?0:.5;}
  const before=frames.slice(),d=applySZShapeConfidence(dictionary,frames,reference,regions);
  assert.ok(d.contactLimitedFrames>0);
  for(let f=0;f<60;f++)assert.equal(frames[f*CHANNELS+jaw]<=.1,before[f*CHANNELS+jaw]<=.1);
  assert.throws(()=>applySZShapeConfidence(dictionary,frames,reference.subarray(CHANNELS),regions),/different duration/);
});

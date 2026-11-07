import {test} from 'node:test';
import assert from 'node:assert/strict';
import channels from '../src/lib/studio/channels.json' with {type:'json'};
import {CHANNELS,FPS} from '../src/lib/dictionary/index.mjs';
import {restEase,neutralContactProtected,applySmoothNeutral} from '../src/lib/dictionary/neutral-transition.mjs';
const jaw=channels.indexOf('jawOpen');

function fixture({end=.6,next='AH',tailOnly=false}={}) {
  const frames=new Float32Array(60*CHANNELS);
  const first={phone:'AH',start:.05,end:.2},last={phone:next,start:end,end:end+.1};
  const piece={};
  const segments=[{start:0,end:.05,pieces:[]},{start:.05,end:.2,pieces:[piece]},
    {start:.2,end:tailOnly?1:end,pieces:[]}];
  if(!tailOnly)segments.push({start:end,end:end+.1,pieces:[piece]},{start:end+.1,end:1,pieces:[]});
  for(let f=0;f<60;f++) {
    const t=f/FPS;
    frames[f*CHANNELS+jaw]=t<=.2?.8:!tailOnly&&t>=end&&t<=end+.1?.6:0;
  }
  const lipRig={neutral:[0],deltas:Array.from({length:CHANNELS},(_,c)=>[c===jaw?1:0])};
  return {dictionary:{lipRig},frames,plan:{segments,phones:tailOnly?[first]:[first,last]}};
}

test('rest envelope is bounded and monotonic with flat velocity and acceleration at its ends',()=>{
  assert.equal(restEase(-1),0);assert.equal(restEase(2),1);assert.equal(restEase(0),0);assert.equal(restEase(1),1);
  let previous=0;
  for(let i=0;i<=10000;i++){const value=restEase(i/10000);assert.ok(value>=0&&value<=1&&value>=previous-1e-14);previous=value;}
  assert.ok(restEase(.001)<1e-7);assert.ok(1-restEase(.999)<1e-7);
});

test('tail relaxes from the actual baked speech pose over 200ms and reaches zero',()=>{
  const {dictionary,frames,plan}=fixture({tailOnly:true}),before=frames.slice();
  const d=applySmoothNeutral(dictionary,frames,plan);
  assert.equal(d.intervals[0].fadeSeconds,.2);assert.equal(d.intervals[0].kind,'tail');
  assert.equal(frames[12*CHANNELS+jaw],before[12*CHANNELS+jaw]); // Exact boundary untouched.
  assert.ok(Math.abs(frames[18*CHANNELS+jaw]-.4)<1e-6);
  assert.ok(frames[15*CHANNELS+jaw]>.6); // Still relaxing at 50ms.
  assert.equal(frames[24*CHANNELS+jaw],0);
  assert.deepEqual(frames.subarray(0,13*CHANNELS),before.subarray(0,13*CHANNELS));
});

test('short internal fades share the available quiet interval without overlapping or overshooting',()=>{
  const {dictionary,frames,plan}=fixture({end:.32});
  const d=applySmoothNeutral(dictionary,frames,plan);
  assert.ok(Math.abs(d.intervals[0].fadeSeconds-.06)<1e-9);
  assert.ok(frames.every(v=>Number.isFinite(v)&&v>=0&&v<=1));
  assert.equal(d.rejectedContactEpisodes,0);
});

test('leading preparation and accepted recorded pauses stay bit-identical',()=>{
  const {dictionary,frames,plan}=fixture(),before=frames.slice();
  plan.pauseDiagnostics={transitions:[{start:.2,end:.6}]};
  const d=applySmoothNeutral(dictionary,frames,plan);
  assert.equal(d.recordedPausesSkipped,1);
  assert.deepEqual(frames.subarray(0,42*CHANNELS),before.subarray(0,42*CHANNELS));
  assert.equal(d.intervals.length,1);assert.equal(d.intervals[0].kind,'tail');
});

test('an upcoming P/B/M keeps every anticipatory contact frame while resting timing may move',()=>{
  for(const next of ['P','B','M']) {
    const {dictionary,frames,plan}=fixture({next}),before=frames.slice();
    const d=applySmoothNeutral(dictionary,frames,plan);
    assert.ok(d.contactLimitedFrames>0);
    for(let f=0;f<60;f++)if(neutralContactProtected(plan.phones,f/FPS))assert.equal(frames[f*CHANNELS+jaw]<=.1,before[f*CHANNELS+jaw]<=.1);
  }
});

test('a truncated tail cannot introduce a new contact episode',()=>{
  const {dictionary,frames,plan}=fixture({tailOnly:true});
  plan.segments[2].end=.22;
  const short=frames.slice(0,14*CHANNELS);short[13*CHANNELS+jaw]=.3;
  const before=short.slice(),d=applySmoothNeutral(dictionary,short,plan);
  assert.equal(d.rejectedContactEpisodes,1);assert.deepEqual(short,before);
});

test('invalid durations are rejected and missing geometry remains a bounded fade',()=>{
  const {dictionary,frames,plan}=fixture({tailOnly:true});
  assert.throws(()=>applySmoothNeutral(dictionary,frames,plan,{seconds:0}),RangeError);
  assert.throws(()=>applySmoothNeutral(dictionary,frames,plan,{seconds:NaN}),RangeError);
  delete dictionary.lipRig;
  assert.ok(applySmoothNeutral(dictionary,frames,plan).changedFrames>0);
  assert.ok(frames.every(v=>Number.isFinite(v)&&v>=0&&v<=1));
});

test('an uncovered dictionary interval inside speech is not treated as a pause',()=>{
  const {dictionary,frames,plan}=fixture(),before=frames.slice();
  plan.segments[2].kind='miss';
  const d=applySmoothNeutral(dictionary,frames,plan);
  assert.deepEqual(frames.subarray(0,42*CHANNELS),before.subarray(0,42*CHANNELS));
  assert.equal(d.intervals.length,1);assert.equal(d.intervals[0].kind,'tail');
});

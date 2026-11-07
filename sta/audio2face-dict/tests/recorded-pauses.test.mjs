import { test } from 'node:test';
import assert from 'node:assert/strict';
import channels from '../src/lib/studio/channels.json' with { type:'json' };
import { applyRecordedPauses, speechContactProtected } from '../src/lib/dictionary/recorded-pauses.mjs';
import { planAnimation,bakeAnimation } from '../src/lib/dictionary/animate.mjs';
import { buildIndex,CHANNELS,FPS } from '../src/lib/dictionary/index.mjs';

const jaw=channels.indexOf('jawOpen'), close=channels.indexOf('mouthClose');
const gap=(frames,f)=>frames[f*CHANNELS+jaw]-frames[f*CHANNELS+close];
function fixture(next='T') {
  const phones=[{phone:'AH',start:.05,end:.1},{phone:'N',start:.1,end:.2},{phone:next,start:.6,end:.7},{phone:'EH',start:.7,end:.8}];
  const runs=[{from:0,to:2,pause:{start:.2,end:.6}},{from:2,to:4}];
  const weights=new Float32Array(60*CHANNELS),frames=new Float32Array(60*CHANNELS);
  for(let f=0;f<60;f++) {
    weights[f*CHANNELS+jaw]=f<12?.5:f<18?.5-(f-12)*.4/6:f<30?.1:.1+Math.min(1,(f-30)/6)*.4;
    weights[f*CHANNELS+close]=.2;
    frames[f*CHANNELS+jaw]=f<=12||f>=36?.5:f===13?.25:0;
    frames[f*CHANNELS+channels.indexOf('eyeBlinkLeft')]=.25;
  }
  const lipRig={neutral:[0],deltas:Array.from({length:CHANNELS},(_,i)=>[i===jaw?1:i===close?-1:0])};
  const clips=[{id:'internal-pause',phones,runs,frameOffset:0,frameCount:60}];
  return {dictionary:{clips,weights,lipRig,index:buildIndex(clips)},plan:{phones,runs},frames};
}

test('recorded release and preparation replace neutral while leaving speech and other channels unchanged',()=>{
  const {dictionary,plan,frames}=fixture();const original=frames.slice();
  const d=applyRecordedPauses(dictionary,frames,plan);
  assert.equal(d.detectedPauses,1);assert.equal(d.usedPauses,1);assert.ok(d.changedFrames>0);
  assert.equal(d.transitions[0].outgoing.pair,true);assert.equal(d.transitions[0].incoming.pair,true);
  for(let f=0;f<60;f++)for(let c=0;c<CHANNELS;c++) {
    const at=f*CHANNELS+c;
    assert.ok(Number.isFinite(frames[at])&&frames[at]>=0&&frames[at]<=1);
    if(f/FPS<.2||f/FPS>=.6||!/^mouth|^jaw/.test(channels[c]))assert.equal(frames[at],original[at]);
  }
  assert.ok(d.transitions[0].outgoing.to<.6);assert.ok(d.transitions[0].incoming.from>.2);
});

test('P/B/M advance windows preserve every original speech-contact frame',()=>{
  for(const phone of ['P','B','M']) {
    const {dictionary,plan,frames}=fixture(phone),original=frames.slice();
    const d=applyRecordedPauses(dictionary,frames,plan);
    assert.equal(d.usedPauses,1);
    for(let f=0;f<60;f++)if(speechContactProtected(plan.phones,f/FPS))assert.equal(gap(frames,f)<=.1,gap(original,f)<=.1);
  }
});

test('missing pause metadata, unsupported phones and missing lip geometry fall back unchanged',()=>{
  for(const change of [x=>{delete x.dictionary.clips[0].runs[0].pause;},x=>{x.plan.phones=x.plan.phones.map(p=>({...p,phone:'HH'}));},x=>{delete x.dictionary.lipRig;}]) {
    const x=fixture(),original=x.frames.slice();change(x);
    assert.equal(applyRecordedPauses(x.dictionary,x.frames,x.plan).usedPauses,0);
    assert.deepEqual(x.frames,original);
  }
});

test('a new isolated quiet contact cycle rejects the transition and restores every frame',()=>{
  const {dictionary,plan,frames}=fixture();
  for(let f=0;f<60;f++)frames[f*CHANNELS+jaw]=.5;
  const original=frames.slice(),d=applyRecordedPauses(dictionary,frames,plan);
  assert.equal(d.usedPauses,0);assert.equal(d.rejectedContactCycles,1);assert.deepEqual(frames,original);
});

test('old bundles and a plan without internal pauses do not affect playback',()=>{
  const {dictionary,plan,frames}=fixture(),original=frames.slice();plan.runs=[{from:0,to:4}];
  const d=applyRecordedPauses(dictionary,frames,plan);
  assert.equal(d.detectedPauses,0);assert.deepEqual(frames,original);
});

test('frame counts use integer sample arithmetic at exact 60fps boundaries',()=>{
  const {dictionary,plan}=fixture();
  const pcm=new Float32Array(66400).fill(.05);
  const result=planAnimation(dictionary,plan.phones,pcm,{closureAware:false,joinAware:false,openingContext:false,boundedContext:false});
  assert.equal(result.frameCount,249);assert.equal(bakeAnimation(dictionary,result).length,249*CHANNELS);
});

import {test} from 'node:test';
import assert from 'node:assert/strict';
import {planSpeech} from '../src/lib/speech-plan.ts';
const bank=Array.from({length:7},(_,i)=>({id:`talk-${i+3}`,kind:'talking' as const,seconds:i+3}));
const lengths=(duration:number)=>planSpeech(duration,bank,()=>0).map(c=>c.seconds);
test('ceil-to-fit selects the closest single clip and balanced pairs or longer sequences',()=>{
 assert.deepEqual(lengths(1.2),[3]);assert.deepEqual(lengths(6),[6]);assert.deepEqual(lengths(6.4),[7]);assert.deepEqual(lengths(9),[9]);
 assert.deepEqual(lengths(9.01),[5,5]);assert.deepEqual(lengths(12.4),[6,7]);assert.deepEqual(lengths(13.4),[7,7]);assert.deepEqual(lengths(18),[9,9]);assert.deepEqual(lengths(18.01),[6,6,7]);assert.deepEqual(lengths(29.2),[7,7,8,8]);
 for(let d=.05;d<100;d+=.17){const plan=lengths(d);assert.equal(plan.length,Math.ceil(Math.ceil(d)/9));assert.ok(plan.reduce((a,b)=>a+b,0)>=d);assert.ok(Math.max(...plan)-Math.min(...plan)<=1)}
});
test('idle clips are excluded, equivalent durations can vary, and legacy talking clips remain playable',()=>{
 assert.equal(planSpeech(6.4,[...bank,{id:'idle',kind:'idle',seconds:7}],()=>0)[0].id,'talk-7');
 assert.equal(planSpeech(6.4,[...bank,{id:'alternate',kind:'talking',seconds:7}],()=>.99)[0].id,'alternate');
 assert.deepEqual(planSpeech(13,[{id:'legacy',seconds:10}]).map(c=>c.seconds),[10,10]);
 assert.throws(()=>planSpeech(1,[]));assert.throws(()=>planSpeech(NaN,bank));
});

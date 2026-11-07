import { test } from 'node:test';
import assert from 'node:assert/strict';
import { planAnimation, bakeAnimation } from '../src/lib/dictionary/animate.mjs';
import { buildIndex, CHANNELS } from '../src/lib/dictionary/index.mjs';
import { VISEME_PAIRS, ALL_VISEME_PAIRS, substitutionMap } from '../src/lib/dictionary/viseme-pairs.mjs';

const phones = (labels) => labels.map((phone, i) => ({ phone, start: 0.1 + i * 0.1, end: 0.2 + i * 0.1 }));
const dictionary = (sequences) => {
  const clips = sequences.map((labels, i) => ({ id: 'clip-' + i, phones: phones(labels), frameCount: 60, frameOffset: i * 60, runs: [{ from: 0, to: labels.length }] }));
  return { clips, index: buildIndex(clips), weights: new Float32Array(clips.length * 60 * CHANNELS) };
};

for (const [from, to] of Object.values(VISEME_PAIRS).flatMap(([a,b])=>[[a,b],[b,a]])) {
  for (const position of [0, 1, 2]) {
    test(`missing triple supports ${from} to ${to} at position ${position}`, () => {
      const source = ['AA', 'IY', 'UW']; source[position] = to;
      const target = [...source]; target[position] = from;
      const dict = dictionary([source]);
      const targetPhones = phones(target);
      const original = structuredClone(targetPhones);
      const pcm = new Float32Array(9600);
      const baseline = planAnimation(dict, targetPhones, pcm, { visemeFallback: false });
      const plan = planAnimation(dict, targetPhones, pcm, { visemeFallback: true, visemePairs:ALL_VISEME_PAIRS });
      assert.equal(baseline.usage.approximate, 0);
      assert.equal(plan.usage.approximate, 1);
      assert.equal(plan.coverage.approximate, 2);
      assert.equal(plan.approximateMatches[0].from, from);
      assert.equal(plan.approximateMatches[0].to, to);
      assert.ok(ALL_VISEME_PAIRS.includes(plan.approximateMatches[0].pair));
      assert.deepEqual(targetPhones, original, 'Target phonemes and timings must remain unchanged');
      assert.deepEqual(dict.clips[0].phones.map(({ phone }) => phone), source);
      assert.ok(bakeAnimation(dict, plan).every(Number.isFinite));
    });
  }
}

test('exact triples always take priority over P/B alternatives', () => {
  const dict = dictionary([['AA', 'B', 'IY'], ['AA', 'P', 'IY']]);
  const plan = planAnimation(dict, phones(['AA', 'P', 'IY']), new Float32Array(9600), { visemeFallback: true });
  assert.equal(plan.usage.approximate, 0);
  assert.equal(plan.usage.triple, 1);
  assert.equal(plan.segments.find(({ kind }) => kind === 'triple').pieces[0].match.clip.id, 'clip-1');
});

for(const [pair,[from,to]] of Object.entries(VISEME_PAIRS)) {
  test(`exact triples take priority with ${pair} enabled`,()=>{
    const dict=dictionary([['AA',to,'IY'],['AA',from,'IY']]);
    const p=planAnimation(dict,phones(['AA',from,'IY']),new Float32Array(9600),{visemePairs:ALL_VISEME_PAIRS});
    assert.equal(p.usage.approximate,0);assert.equal(p.usage.triple,1);
  });
  test(`${pair} cannot bridge a confirmed silence`,()=>{
    const target=phones(['AA',from,'IY']);target[2].start=.7;target[2].end=.8;
    const p=planAnimation(dictionary([['AA',to,'IY']]),target,new Float32Array(16000),{visemePairs:ALL_VISEME_PAIRS});
    assert.equal(p.usage.approximate,0);
  });
}

test('additional pairs are opt-in and can be tested individually',()=>{
  for(const [pair,[from,to]] of Object.entries(VISEME_PAIRS)) {
    if(pair==='P/B')continue;
    const dict=dictionary([['AA',to,'IY']]),target=phones(['AA',from,'IY']),pcm=new Float32Array(9600);
    assert.equal(planAnimation(dict,target,pcm).usage.approximate,0);
    assert.equal(planAnimation(dict,target,pcm,{visemePairs:['P/B',pair]}).usage.approximate,1);
  }
});

test('expanded fallback still permits only one substitution and excludes other viseme groups',()=>{
  for(const [source,target] of [
    [['V','Z','AA'],['F','S','AA']],
    [['V','V','AA'],['F','F','AA']],
    [['AA','M','IY'],['AA','B','IY']],
    [['AA','W','IY'],['AA','R','IY']],
    [['AA','JH','IY'],['AA','CH','IY']],
    [['AA','AE','IY'],['AA','AH','IY']],
  ])assert.equal(planAnimation(dictionary([source]),phones(target),new Float32Array(9600),{visemePairs:ALL_VISEME_PAIRS}).usage.approximate,0);
});

test('unknown substitution pairs are rejected and an empty list disables approximate matches',()=>{
  assert.throws(()=>substitutionMap(['M/P']),RangeError);
  assert.throws(()=>substitutionMap('F/V'),TypeError);
  const p=planAnimation(dictionary([['AA','B','IY']]),phones(['AA','P','IY']),new Float32Array(9600),{visemePairs:[]});
  assert.equal(p.usage.approximate,0);
});

test('only one P/B substitution is allowed; M and unrelated phones are excluded', () => {
  for (const [source, target] of [
    [['B', 'B', 'AA'], ['P', 'P', 'AA']],
    [['AA', 'M', 'IY'], ['AA', 'P', 'IY']],
    [['AA', 'F', 'IY'], ['AA', 'P', 'IY']],
  ]) {
    const plan = planAnimation(dictionary([source]), phones(target), new Float32Array(9600), { visemeFallback: true });
    assert.equal(plan.usage.approximate, 0);
  }
});

test('well-timed exact doubles can beat a poorly timed approximate triple', () => {
  const dict = dictionary([['AA', 'P'], ['P', 'IY'], ['AA', 'B', 'IY']]);
  dict.clips[2].phones = dict.clips[2].phones.map((phone, i) => ({ ...phone, start: 0.1 + i * 0.3, end: 0.4 + i * 0.3 }));
  const plan = planAnimation(dict, phones(['AA', 'P', 'IY']), new Float32Array(16000), { visemeFallback: true });
  assert.equal(plan.usage.approximate, 0);
  assert.equal(plan.usage.pair, 2);
});

test('approximate triples do not bridge a confirmed silence', () => {
  const target = phones(['AA', 'P', 'IY']);
  target[2].start = 0.7; target[2].end = 0.8;
  const plan = planAnimation(dictionary([['AA', 'B', 'IY']]), target, new Float32Array(16000), { visemeFallback: true });
  assert.equal(plan.usage.approximate, 0);
});

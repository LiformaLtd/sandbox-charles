import { test } from 'node:test';
import assert from 'node:assert/strict';
import { planAnimation, bakeAnimation } from '../src/lib/dictionary/animate.mjs';
import { buildIndex, CHANNELS } from '../src/lib/dictionary/index.mjs';

const phones = (labels) => labels.map((phone, i) => ({ phone, start: 0.1 + i * 0.1, end: 0.2 + i * 0.1 }));
const dictionary = (sequences) => {
  const clips = sequences.map((labels, i) => ({ id: 'clip-' + i, phones: phones(labels), frameCount: 60, frameOffset: i * 60, runs: [{ from: 0, to: labels.length }] }));
  return { clips, index: buildIndex(clips), weights: new Float32Array(clips.length * 60 * CHANNELS) };
};

for (const [from, to] of [['P', 'B'], ['B', 'P']]) {
  for (const position of [0, 1, 2]) {
    test(`missing triple supports ${from} to ${to} at position ${position}`, () => {
      const source = ['AA', 'IY', 'UW']; source[position] = to;
      const target = [...source]; target[position] = from;
      const dict = dictionary([source]);
      const targetPhones = phones(target);
      const original = structuredClone(targetPhones);
      const pcm = new Float32Array(9600);
      const baseline = planAnimation(dict, targetPhones, pcm, { visemeFallback: false });
      const plan = planAnimation(dict, targetPhones, pcm, { visemeFallback: true });
      assert.equal(baseline.usage.approximate, 0);
      assert.equal(plan.usage.approximate, 1);
      assert.equal(plan.coverage.approximate, 2);
      assert.equal(plan.approximateMatches[0].from, from);
      assert.equal(plan.approximateMatches[0].to, to);
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

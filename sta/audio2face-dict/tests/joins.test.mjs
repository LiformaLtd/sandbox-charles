import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildIndex, CHANNELS, FPS, speechRuns } from '../src/lib/dictionary/index.mjs';
import { closureBlend, CONTACT_GAP } from '../src/lib/dictionary/closures.mjs';
import { bakeAnimation, joinSourceTimes, planAnimation } from '../src/lib/dictionary/animate.mjs';

const phones = (labels, start = 0.05, duration = 0.1) => labels.map((phone, i) => ({
  phone, start: start + duration * i, end: start + duration * (i + 1),
}));

function dictionary(labels, duration = 0.1) {
  const clips = labels.map((sequence, i) => ({
    id: 'clip-' + i, frameOffset: i * 60, frameCount: 60,
    phones: phones(sequence, 0.05, duration), runs: [{ from: 0, to: sequence.length }],
  }));
  const weights = new Float32Array(clips.length * 60 * CHANNELS);
  for (const [i, clip] of clips.entries()) {
    for (let f = 0; f < 60; f++) weights[(clip.frameOffset + f) * CHANNELS] = i * 0.4 + f / 120;
  }
  return { clips, weights, index: buildIndex(clips) };
}

for (const [name, tracks, target] of [
  ['double to double', [['A', 'B'], ['B', 'C']], ['A', 'B', 'C']],
  ['triple to double', [['A', 'B', 'C'], ['C', 'D']], ['A', 'B', 'C', 'D']],
  ['double to triple', [['A', 'B'], ['B', 'C', 'D']], ['A', 'B', 'C', 'D']],
]) {
  test(name + ' uses moving frames beyond both nominal sample spans', () => {
    const dict = dictionary(tracks);
    const plan = planAnimation(dict, phones(target), new Float32Array(9000));
    assert.equal(plan.joins.length, 1);
    const join = plan.joins[0];
    assert.ok(joinSourceTimes(join, join.anchor - 0.01)[1] < join.incoming.from);
    assert.ok(joinSourceTimes(join, join.anchor + 0.01)[0] > join.outgoing.to);
    const frames = bakeAnimation(dict, plan);
    const trackValue = (item, time) => Number(item.match.clip.id.at(-1)) * 0.4 + time / 2;
    const first = Math.ceil(join.start * FPS);
    const last = Math.floor(join.end * FPS);
    assert.ok(last > first);
    for (let frame = first; frame <= last; frame++) {
      const time = frame / FPS;
      const [out, incoming] = joinSourceTimes(join, time);
      const amount = (time - join.start) / (join.end - join.start);
      const expected = trackValue(join.outgoing, out) * (1 - amount) + trackValue(join.incoming, incoming) * amount;
      assert.ok(Math.abs(frames[frame * CHANNELS] - expected) < 1e-6);
    }
  });
}

test('a crossfade between frame timestamps contributes a mixed output frame', () => {
  const dict = dictionary([['A', 'B'], ['B', 'C']], 0.02);
  const plan = planAnimation(dict, phones(['A', 'B', 'C'], 0.045, 0.02), new Float32Array(4000));
  const [join] = plan.joins;
  assert.ok((Math.floor(join.start * FPS) + 1) / FPS >= join.end);
  const frames = bakeAnimation(dict, plan);
  const unblended = bakeAnimation(dict, { ...plan, joins: [] });
  const before = Math.floor(join.anchor * FPS) * CHANNELS;
  const after = Math.ceil(join.anchor * FPS) * CHANNELS;
  assert.ok(frames[before] > unblended[before] + 0.02);
  assert.ok(frames[after] < unblended[after] - 0.02);
});

test('real silence splits a run but audible PCM in a long alignment gap does not', () => {
  const list = [
    { phone: 'A', start: 0.04, end: 0.08 },
    { phone: 'B', start: 0.32, end: 0.36 },
  ];
  const pcm = new Float32Array(6400);
  assert.equal(speechRuns(list, pcm).length, 2);
  pcm.fill(0.04, 1280, 5120);
  assert.equal(speechRuns(list, pcm).length, 1);
});

test('single phones prefer isolated matches and record misses', () => {
  const dict = dictionary([['A', 'B'], ['A']]);
  const pcm = new Float32Array(4000);
  const matched = planAnimation(dict, phones(['A']), pcm);
  assert.equal(matched.coverage.single, 1);
  assert.equal(matched.segments[1].pieces[0].match.clip.id, 'clip-1');
  const missing = planAnimation(dict, phones(['Z']), pcm);
  assert.equal(missing.coverage.miss, 1);
  assert.ok(bakeAnimation(dict, missing).every((weight) => weight === 0));
});

test('single-phoneme fallback fills only missing pairs and reuses the surrounding exact poses', () => {
  const dict = dictionary([['A', 'B'], ['C', 'D']]);
  const target = phones(['A', 'B', 'C', 'D']);
  const pcm = new Float32Array(9000);
  const exact = planAnimation(dict, target, pcm, { singleFallback: false });
  const plan = planAnimation(dict, target, pcm);
  assert.deepEqual(exact.coverage, { triple: 0, pair: 2, single: 0, fallback: 0, miss: 1 });
  assert.deepEqual(plan.coverage, { triple: 0, pair: 2, single: 0, fallback: 1, miss: 0 });
  const frames = bakeAnimation(dict, plan);
  const old = bakeAnimation(dict, exact);
  const [gap] = plan.segments.filter((part) => part.kind === 'fallback');
  assert.equal(plan.joins.length, 0);
  assert.ok(Math.abs(frames[12 * CHANNELS] - old[12 * CHANNELS]) < 1e-6);
  assert.ok(Math.abs(frames[18 * CHANNELS] - old[18 * CHANNELS]) < 1e-6);
  // Halfway from B to C, sample the end of source B and the start of source C.
  assert.ok(Math.abs(frames[15 * CHANNELS] - ((0.25 / 2) + (0.4 + 0.05 / 2)) / 2) < 1e-6);
  assert.equal(gap.pieces[0].match.clip.id, 'clip-0');
  assert.equal(gap.pieces[1].match.clip.id, 'clip-1');
});

test('a missing singleton leaves the interval uncovered', () => {
  const dict = dictionary([['A']]);
  const plan = planAnimation(dict, phones(['A', 'Z']), new Float32Array(9000));
  assert.equal(plan.coverage.miss, 1);
  assert.equal(plan.coverage.fallback, 0);
  assert.ok(bakeAnimation(dict, plan).every((weight) => weight === 0));
});

test('join-aware matching prefers compatible motion over a slightly closer timing match', () => {
  const dict = dictionary([['A', 'B'], ['B', 'C'], ['B', 'C']]);
  dict.clips[2].phones[1].end += 0.02;
  for (let frame = 0; frame < 60; frame++) {
    const jawOpen = 17;
    dict.weights[(60 + frame) * CHANNELS + jawOpen] = 0.8;
  }
  const target = phones(['A', 'B', 'C']);
  const pcm = new Float32Array(9000);
  const baseline = planAnimation(dict, target, pcm, { joinAware: false });
  const joined = planAnimation(dict, target, pcm);
  assert.equal(baseline.joins[0].inClip, 'clip-1');
  assert.equal(joined.joins[0].inClip, 'clip-2');
  assert.deepEqual(joined.coverage, baseline.coverage);
  const aroundJoin = (plan) => {
    const frames = bakeAnimation(dict, plan);
    return Math.abs(frames[13 * CHANNELS + 17] - frames[12 * CHANNELS + 17]);
  };
  assert.ok(aroundJoin(joined) < aroundJoin(baseline));
});

test('lip gap features can distinguish candidates with identical pose proxies', () => {
  const dict = dictionary([['A', 'B'], ['B', 'C'], ['B', 'C']]);
  dict.clips[2].phones[1].end += 0.02;
  const mouthPress = 36;
  for (let frame = 0; frame < 60; frame++) {
    dict.weights[(60 + frame) * CHANNELS + mouthPress] = 0.7;
  }
  const target = phones(['A', 'B', 'C']);
  const pcm = new Float32Array(9000);
  assert.equal(planAnimation(dict, target, pcm).joins[0].inClip, 'clip-1');

  dict.lipRig = {
    deltas: Array.from({ length: CHANNELS }, (_, channel) =>
      Array(9).fill(channel === mouthPress ? -2 : 0)),
  };
  const plan = planAnimation(dict, target, pcm);
  assert.equal(plan.joins[0].inClip, 'clip-2');
  assert.equal(plan.coverage.pair, 2);
});

test('closure blending requires audio evidence and preserves crossfade endpoints', () => {
  const left = { gap: 0.03 };
  const right = { gap: 0.24 };
  const guide = { probability: [1], gapAt: (clip) => clip.gap };
  const blend = (weight) => closureBlend(guide, 0, left, 0, right, 0, weight);
  const gap = (weight) => left.gap + weight * (right.gap - left.gap);
  assert.ok(gap(0.5) > CONTACT_GAP);
  assert.ok(gap(blend(0.5)) < CONTACT_GAP);
  assert.equal(blend(0), 0);
  assert.equal(blend(1), 1);
  guide.probability[0] = 0.5;
  assert.equal(blend(0.5), 0.5);
  guide.probability[0] = 0;
  assert.ok(gap(0.2) < CONTACT_GAP);
  assert.ok(gap(blend(0.2)) > CONTACT_GAP);
});

test('match counts count a triple once and include single-phoneme fallback sources', () => {
  const pcm = new Float32Array(9000);
  const triple = planAnimation(dictionary([['A', 'B', 'C']]), phones(['A', 'B', 'C']), pcm);
  assert.equal(triple.coverage.triple, 2);
  assert.deepEqual(triple.usage, { triple: 1, pair: 0, single: 0 });
  const fallback = planAnimation(dictionary([['A', 'B'], ['C', 'D']]), phones(['A', 'B', 'C', 'D']), pcm);
  assert.deepEqual(fallback.usage, { triple: 0, pair: 2, single: 2 });
  const isolated = planAnimation(dictionary([['A']]), phones(['A']), pcm);
  assert.deepEqual(isolated.usage, { triple: 0, pair: 0, single: 1 });
});

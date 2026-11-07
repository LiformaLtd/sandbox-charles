import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeOpeningContext, openingSourceTime } from '../src/lib/dictionary/opening-context.mjs';
import { planAnimation, bakeAnimation } from '../src/lib/dictionary/animate.mjs';
import { buildIndex, CHANNELS, FPS } from '../src/lib/dictionary/index.mjs';

function fixture() {
  const source = [{ phone: 'AH', start: .5, end: .52 }, { phone: 'B', start: .6, end: .62 }, { phone: 'R', start: .64, end: .68 }];
  const target = [{ phone: 'AH', start: .15, end: .17 }, { phone: 'B', start: .31, end: .33 }, { phone: 'R', start: .35, end: .39 }];
  const clips = [{ id: 'donor', phones: source, frameOffset: 0, frameCount: 60, runs: [{ from: 0, to: 3 }] }];
  const weights = new Float32Array(60 * CHANNELS);
  for (let f = 0; f < 60; f++) weights[f * CHANNELS] = f < 27 ? .55 : .2;
  const lipRig = { neutral: Array(9).fill(-.15), deltas: Array.from({ length: CHANNELS }, (_, c) => Array(9).fill(c === 0 ? 1 : 0)) };
  return { dictionary: { clips, weights, lipRig, index: buildIndex(clips) }, target, pcm: new Float32Array(16000).fill(.05) };
}

test('opening context recovers an actual open donor pose and leaves the body bit-identical', () => {
  const { dictionary, target, pcm } = fixture();
  const beforePlan = planAnimation(dictionary, target, pcm, { openingContext: false });
  const afterPlan = planAnimation(dictionary, target, pcm, { openingContext: true });
  assert.ok(afterPlan.leadingContext);
  const before = bakeAnimation(dictionary, beforePlan), after = bakeAnimation(dictionary, afterPlan);
  assert.ok(-.15 + before[0] <= .1);
  assert.ok(-.15 + after[0] > .1);
  assert.deepEqual(after.subarray(Math.ceil(target[0].start * FPS) * CHANNELS), before.subarray(Math.ceil(target[0].start * FPS) * CHANNELS));
  assert.deepEqual(afterPlan.coverage, beforePlan.coverage);
  assert.deepEqual(afterPlan.usage, beforePlan.usage);
  assert.equal(after.length, before.length);
  const context = afterPlan.leadingContext;
  assert.equal(openingSourceTime(context, -1), context.sourceFrom);
  assert.equal(openingSourceTime(context, 1), context.sourceTo);
  assert.ok(context.sourceFrom >= 0 && context.sourceFrom < context.sourceTo);
  assert.ok((context.sourceTo - context.sourceFrom) / context.targetEnd <= 2);
});

test('lead-in starts after an unrelated earlier donor closure', () => {
  const { dictionary, target, pcm } = fixture();
  for (let f = 18; f <= 22; f++) dictionary.weights[f * CHANNELS] = .2;
  const plan = planAnimation(dictionary, target, pcm, { openingContext: true });
  assert.ok(plan.leadingContext.sourceFrom >= 23 / FPS);
  assert.ok(-.15 + bakeAnimation(dictionary, plan)[0] > .1);
});

test('no lead-in is invented when no recorded opening or no target lead time exists', () => {
  const { dictionary, target, pcm } = fixture();
  for (let f = 0; f < 60; f++) dictionary.weights[f * CHANNELS] = .2;
  assert.equal(planAnimation(dictionary, target, pcm, { openingContext: true }).leadingContext, undefined);
  const normal = fixture();
  const segments = planAnimation(normal.dictionary, normal.target, pcm).segments;
  assert.equal(makeOpeningContext(normal.dictionary, [{ ...target[0], start: 0 }, ...target.slice(1)], segments), undefined);
});

test('initial consonants and already-open vowel samples retain their normal start', () => {
  const { dictionary, target, pcm } = fixture();
  const segments = planAnimation(dictionary, target, pcm).segments;
  assert.equal(makeOpeningContext(dictionary, [{ ...target[0], phone: 'B' }, ...target.slice(1)], segments), undefined);
  assert.equal(makeOpeningContext(dictionary, [target[0], { ...target[1], phone: 'R' }, target[2]], segments), undefined);
  for (let f = 0; f < 60; f++) dictionary.weights[f * CHANNELS] = .55;
  assert.equal(makeOpeningContext(dictionary, target, segments), undefined);
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import channels from '../src/lib/studio/channels.json' with { type: 'json' };
import { CHANNELS, FPS } from '../src/lib/dictionary/index.mjs';
import { boundedContextEnabled, contextSourceTime, contextWeight, makeBoundedContexts, applyBoundedContexts } from '../src/lib/dictionary/bounded-context.mjs';

function fixture() {
  const phones = ['AH','W','F','AH'].map((phone, i) => ({ phone, start: .2 + i * .2, end: .3 + i * .2 }));
  const clip = { phones, frameOffset: 0, frameCount: 72, runs: [{ from: 0, to: 4 }] };
  const match = { clip, source: phones.slice(1, 3), count: 2 };
  const lipRig = { neutral: [0], deltas: Array.from({ length: CHANNELS }, (_, i) => [i === channels.indexOf('mouthClose') ? -1 : i === channels.indexOf('jawOpen') ? 1 : 0]) };
  const weights = new Float32Array(clip.frameCount * CHANNELS).fill(.8);
  const plan = { phones, runs: [{ from: 0, to: 4 }], duration: 1.2,
    segments: [{ first: phones[1], pieces: [{ first: phones[1], match }] }] };
  const entry = (feature) => ({ candidateBeforeMs: 350, candidateAfterMs: 250, evidence: [{ feature }] });
  const candidates = { phonemes: { W: entry('rounding'), F: entry('fvProximity') } };
  return { dictionary: { lipRig, weights }, plan, candidates, clip };
}

test('custom margins use real before/after frames and stop at adjacent midpoints', () => {
  const { dictionary, plan, candidates } = fixture();
  const contexts = makeBoundedContexts(dictionary, plan, { candidates });
  assert.equal(contexts.length, 2);
  assert.equal(contexts[0].start, .25);
  assert.equal(contexts[0].end, .45);
  assert.ok(contexts[1].end <= .850001);
  assert.ok(contextSourceTime(contexts[0], contexts[0].start) < plan.phones[1].start);
  assert.ok(contextSourceTime(contexts[1], contexts[1].end) > plan.phones[2].end);
  assert.deepEqual(contexts[0].poseChannels.map(i => channels[i]).includes('jawOpen'), false);
  assert.equal(contextWeight(contexts[0], contexts[0].start), 0);
  assert.equal(contextWeight(contexts[0], contexts[0].end), 0);
  assert.equal(contextWeight(contexts[0], -1), 0);
});

test('narrow measured margins override broader neighboring context', () => {
  const { dictionary, plan, candidates } = fixture();
  candidates.phonemes.W.candidateBeforeMs = 10;
  candidates.phonemes.F.candidateAfterMs = 10;
  const [before, after] = makeBoundedContexts(dictionary, plan, { candidates });
  assert.ok(Math.abs(before.start - .39) < 1e-9);
  assert.ok(Math.abs(after.end - .71) < 1e-9);
});

test('silence, donor run boundaries and clip end bound the contributions', () => {
  const { dictionary, plan, candidates, clip } = fixture();
  plan.runs = [{ from: 0, to: 1, pause: { start: .3, end: .38 } }, { from: 1, to: 3, pause: { start: .75, end: .8 } }];
  clip.runs = [{ from: 0, to: 3 }, { from: 3, to: 4 }];
  const contexts = makeBoundedContexts(dictionary, plan, { candidates });
  assert.equal(contexts.length, 1); // No prefix across the preceding silence.
  assert.ok(contexts[0].end <= plan.phones[2].end);
  clip.runs = [{ from: 0, to: 4 }];
  plan.runs = [{ from: 0, to: 4 }];
  clip.frameCount = 44;
  for (const context of makeBoundedContexts(dictionary, plan, { candidates })) {
    assert.ok(contextSourceTime(context, context.end) <= (clip.frameCount - 1) / FPS + 1e-9);
  }
});

test('unsupported phones and utterance openings do not invent context', () => {
  const { dictionary, plan, candidates } = fixture();
  assert.deepEqual(makeBoundedContexts(dictionary, plan, { candidates: { phonemes: {} } }), []);
  plan.segments[0].first = plan.phones[0];
  plan.segments[0].pieces[0].first = plan.phones[1];
  assert.equal(makeBoundedContexts(dictionary, plan, { candidates }).some(c => c.side === 'before'), false);
  assert.equal(boundedContextEnabled, false); // Mixed benchmark results: opt-in experiment.
  assert.throws(() => makeBoundedContexts(dictionary, plan, { strength: -1 }), RangeError);
  assert.throws(() => makeBoundedContexts(dictionary, plan, { strength: NaN }), RangeError);
});

test('overlapping contributions stay bounded, preserve every contact frame and leave other channels unchanged', () => {
  const { dictionary, plan, candidates } = fixture();
  const contexts = makeBoundedContexts(dictionary, plan, { candidates });
  const close = channels.indexOf('mouthClose'), jaw = channels.indexOf('jawOpen');
  const frames = new Float32Array(72 * CHANNELS);
  for (let f = 0; f < 72; f++) { frames[f * CHANNELS + jaw] = .6; frames[f * CHANNELS + close] = f % 3 ? .7 : .2; }
  const original = frames.slice();
  const diagnostics = applyBoundedContexts(dictionary, frames, [...contexts, ...contexts, ...contexts]);
  assert.ok(diagnostics.framesBlended > 0);
  assert.ok(diagnostics.contactLimitedFrames > 0);
  const changedChannels = new Set(contexts.flatMap(c => c.poseChannels));
  for (let f = 0; f < 72; f++) {
    const at = f * CHANNELS;
    assert.equal(frames[at + jaw] - frames[at + close] <= .1, original[at + jaw] - original[at + close] <= .1);
    for (let c = 0; c < CHANNELS; c++) {
      assert.ok(Number.isFinite(frames[at + c]) && frames[at + c] >= 0 && frames[at + c] <= 1);
      if (!changedChannels.has(c)) assert.equal(frames[at + c], original[at + c]);
    }
  }
});

test('invalid donor sampling is rejected rather than reading another recording', () => {
  const { dictionary, plan, candidates } = fixture();
  const contexts = makeBoundedContexts(dictionary, plan, { candidates });
  contexts[0].sourceAnchor = -10;
  assert.throws(() => applyBoundedContexts(dictionary, new Float32Array(72 * CHANNELS), contexts), /outside donor clip/);
});

test('an unrelated outer phoneme is rejected, while symmetric P/B neighbors remain compatible', () => {
  const { dictionary, plan, candidates, clip } = fixture();
  plan.phones = plan.phones.map(p => ({ ...p }));
  plan.segments[0].first = plan.phones[1];
  plan.phones[0].phone = 'IY';
  assert.equal(makeBoundedContexts(dictionary, plan, { candidates }).some(c => c.side === 'before'), false);
  assert.equal(makeBoundedContexts(dictionary, plan, { candidates, mode: 'absolute' }).length, 2);
  for (const [a,b] of [['P','B'],['B','P']]) {
    plan.phones[0].phone = a; clip.phones[0].phone = b;
    assert.equal(makeBoundedContexts(dictionary, plan, { candidates }).length, 2);
  }
});

test('low training attribution reduces influence without changing the measured bounds', () => {
  const { dictionary, plan, candidates } = fixture();
  candidates.phonemes.W.evidence[0].matchedFraction = .12;
  const [revised] = makeBoundedContexts(dictionary, plan, { candidates });
  const [legacy] = makeBoundedContexts(dictionary, plan, { candidates, mode: 'absolute' });
  assert.equal(revised.strength, legacy.strength * .12);
  assert.equal(revised.start, legacy.start);
  assert.equal(revised.end, legacy.end);
});

test('outside timing follows the outside interval, rather than the selected pair interior', () => {
  const { dictionary, plan, candidates } = fixture();
  plan.phones = plan.phones.map((p,i) => i === 0 ? { ...p, start: .3, end: .4 } : p);
  const [revised] = makeBoundedContexts(dictionary, plan, { candidates });
  const [legacy] = makeBoundedContexts(dictionary, plan, { candidates, mode: 'absolute' });
  assert.ok(Math.abs(revised.slope - 2) < 1e-8);
  assert.ok(Math.abs(legacy.slope - 1) < 1e-8);
});

test('anchor alignment removes a constant donor pose bias but transfers real surrounding motion', () => {
  const { dictionary, plan, candidates } = fixture();
  const context = makeBoundedContexts(dictionary, plan, { candidates })[0];
  const pucker = channels.indexOf('mouthPucker');
  context.poseChannels = [pucker];
  const frames = new Float32Array(72 * CHANNELS);
  for (let f = 0; f < 72; f++) frames[f * CHANNELS + pucker] = .5;
  const aligned = frames.slice(), absolute = frames.slice();
  applyBoundedContexts(dictionary, aligned, [context]);
  applyBoundedContexts(dictionary, absolute, [context], { mode: 'absolute' });
  assert.ok(aligned.every((v,i) => Math.abs(v - frames[i]) < 1e-7));
  assert.ok(absolute.some((v,i) => Math.abs(v - frames[i]) > .01));
  for (let f = 0; f < 72; f++) dictionary.weights[f * CHANNELS + pucker] = f / 72;
  const dynamic = frames.slice();
  applyBoundedContexts(dictionary, dynamic, [context]);
  assert.ok(dynamic.some((v,i) => Math.abs(v - frames[i]) > .001));
  assert.equal(dynamic[Math.round(context.targetAnchor * FPS) * CHANNELS + pucker], .5);
  assert.ok(dynamic.every(v => Number.isFinite(v) && v >= 0 && v <= 1));
});

test('overlapping anchor offsets are computed before mutation and do not depend on context order', () => {
  const { dictionary, plan, candidates } = fixture();
  const contexts = makeBoundedContexts(dictionary, plan, { candidates });
  const pucker = channels.indexOf('mouthPucker');
  for (const c of contexts) c.poseChannels = [pucker];
  contexts[1].start = .35;
  const frames = new Float32Array(72 * CHANNELS).fill(.4);
  for (let f = 0; f < 72; f++) frames[f * CHANNELS + pucker] = f / 72;
  const a = frames.slice(), b = frames.slice();
  applyBoundedContexts(dictionary, a, contexts);
  applyBoundedContexts(dictionary, b, contexts.toReversed());
  assert.ok(a.every((v,i) => Math.abs(v - b[i]) < 1e-7));
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeClosureGuide, closureCandidateCost, closureBlend } from '../src/lib/dictionary/closures.mjs';
import { retainAdvanceContacts } from '../src/lib/dictionary/retain-contacts.mjs';
import { CHANNELS } from '../src/lib/dictionary/index.mjs';
import profile from '../data/closure-windows.json' with { type: 'json' };

test('advance guide samples the trained early window rather than the old phoneme window', () => {
  const dictionary = { closureModel: { bias: 0, mean: Array(20).fill(0), scale: Array(20).fill(1), weights: Array(20).fill(0) },
    lipRig: { neutral: Array(9).fill(0.2), deltas: Array.from({ length: CHANNELS }, () => Array(9).fill(0)) } };
  const phone = { phone: 'P', start: 1, end: 1.02 }, pcm = new Float32Array(32000);
  const early = makeClosureGuide(dictionary, [phone], pcm, true).samples.get(phone);
  const old = makeClosureGuide(dictionary, [phone], pcm).samples.get(phone);
  assert.ok(early.some(({ time }) => time < phone.start - .05));
  assert.ok(early.every(({ time }) => time >= phone.start + profile.phones.P.onsetRangeMs[0] / 1000 && time < phone.end + profile.phones.P.releaseRangeMs[1] / 1000));
  assert.ok(old.every(({ time }) => time >= phone.start - .02));
});

test('advance selection rewards recorded contact at the early acoustic evidence', () => {
  const target = [{ phone: 'AA', start: .2, end: .4 }, { phone: 'P', start: .5, end: .52 }];
  const probability = new Float32Array(60).fill(-1); probability[24] = 1;
  const guide = { advance: true, probability, gapAt: (clip) => clip.gap };
  assert.ok(closureCandidateCost(guide, { gap: .03 }, target, target) < closureCandidateCost(guide, { gap: .4 }, target, target));
});

test('advance blending preserves real donor contact but cannot manufacture a contact from two open donors', () => {
  const guide = { advance: true, probability: [1], gapAt: (clip) => clip.gap };
  const closed = { gap: .03 }, open = { gap: .24 };
  const amount = closureBlend(guide, 0, closed, 0, open, 0, .5);
  assert.ok(closed.gap + amount * (open.gap - closed.gap) <= .1);
  assert.equal(closureBlend(guide, 0, { gap: .12 }, 0, open, 0, .5), .5);
  assert.equal(closureBlend(guide, 0, closed, 0, open, 0, 0), 0);
  assert.equal(closureBlend(guide, 0, closed, 0, open, 0, 1), 1);
  guide.probability[0] = .5;
  assert.equal(closureBlend(guide, 0, closed, 0, open, 0, .5), .5);
});

test('retention restores one lost baseline contact, including a shared contact, without new contact frames', () => {
  const frames = new Float32Array(60 * CHANNELS).fill(.4), baseline = frames.slice();
  for (let f = 20; f <= 24; f++) baseline[f * CHANNELS] = .05;
  baseline[50 * CHANNELS] = .05; // Outside every learned window: do not restore.
  const guide = { profile, phones: [{ phone: 'P', start: .5, end: .52 }, { phone: 'M', start: .52, end: .54 }], frameGap: (track, offset) => track[offset] };
  assert.equal(retainAdvanceContacts(frames, baseline, guide), 1);
  for (let f = 0; f < 60; f++) {
    assert.equal(frames[f * CHANNELS] <= .1, f >= 20 && f <= 24);
    for (let c = 0; c < CHANNELS; c++) assert.ok(frames[f * CHANNELS + c] >= .05 && frames[f * CHANNELS + c] <= .4 + 1e-7);
  }
  assert.equal(retainAdvanceContacts(frames, baseline, guide), 0, 'An existing event must not be duplicated');
});

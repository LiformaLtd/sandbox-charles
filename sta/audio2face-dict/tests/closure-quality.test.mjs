import { test } from 'node:test';
import assert from 'node:assert/strict';
import { checkAdvanceClosures, summarizeAdvanceClosures, advanceWindow, validateClosureProfile } from '../scripts/lib/closure-quality.mjs';

const profile = { version: 1, fps: 60, threshold: 0.1, phones: Object.fromEntries(['P','B','M'].map((phone) =>
  [phone, { examples: 100, onsetRangeMs: [-160, -30], releaseRangeMs: [-80, -1] }])) };
const phone = { phone: 'P', start: 1, end: 1.02 };
const track = (...ranges) => Float32Array.from({ length: 120 }, (_, frame) => ranges.some(([start,end]) => frame / 60 >= start && frame / 60 <= end) ? 0 : 0.2);

test('a naturally early reference closure is recognised without shifting animation', () => {
  const gaps = track([0.9, 0.95]);
  const before = gaps.slice();
  const rows = checkAdvanceClosures([phone], gaps, gaps, profile);
  assert.equal(rows[0].classification, 'tp');
  assert.ok(rows[0].predictedContact.end < phone.start - .02);
  const summary = summarizeAdvanceClosures(rows);
  assert.equal(summary.tp, 1);
  assert.equal(summary.timing.onsetErrorMs.absolute.mean, 0);
  assert.equal(summary.timing.durationErrorMs.absolute.mean, 0);
  assert.deepEqual(gaps, before);
});

test('wider advance windows still reject missing or late contacts and detect extras', () => {
  const expected = track([0.9, 0.95]);
  assert.equal(checkAdvanceClosures([phone], track(), expected, profile)[0].classification, 'fn');
  assert.equal(checkAdvanceClosures([phone], track([1.15, 1.2]), expected, profile)[0].classification, 'fn');
  assert.equal(checkAdvanceClosures([phone], expected, track(), profile)[0].classification, 'fp');
  assert.equal(checkAdvanceClosures([phone], track(), track(), profile)[0].classification, 'tn');
});

test('a shared episode cannot be credited to two phonemes and ambiguity is explicit', () => {
  const phones = [phone, { phone: 'M', start: 1.04, end: 1.08 }];
  const gaps = track([0.9, 0.95]);
  const rows = checkAdvanceClosures(phones, gaps, gaps, profile);
  assert.ok(rows.every((row) => row.classification === 'excluded_reference'));
  const summary = summarizeAdvanceClosures(rows);
  assert.equal(summary.scoredPhonemes, 0);
  assert.equal(summary.excludedReferenceReasons.shared_or_neighbor_ambiguous, 2);
});

test('fragmented prediction and clip-truncated contacts are not counted as successes', () => {
  const expected = track([0.9, 0.95]);
  const rows = checkAdvanceClosures([phone], track([0.85, 0.86], [0.95, 0.96]), expected, profile);
  assert.equal(rows[0].classification, 'excluded_prediction');
  assert.equal(rows[0].predictionStatus, 'multiple_events');
  assert.equal(checkAdvanceClosures([phone], expected, track([0, 0.95]), profile)[0].referenceStatus, 'clipped_event');
  assert.equal(advanceWindow({ ...phone, start: 0.02, end: 0.04 }, profile, 2).clipped, true);
});

test('evaluation rejects invalid calibration and mismatched track durations', () => {
  assert.throws(() => validateClosureProfile({ ...profile, fps: 24 }), /Invalid/);
  assert.throws(() => checkAdvanceClosures([phone], track(), new Float32Array(5), profile), /different durations/);
  const invalid = structuredClone(profile); invalid.phones.P.onsetRangeMs = [-20, -160];
  assert.throws(() => validateClosureProfile(invalid), /Invalid timing ranges/);
});

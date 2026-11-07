import { test } from 'node:test';
import assert from 'node:assert/strict';
import { contactEvents, associateContacts, offsetMeasurements, distribution } from '../scripts/lib/closure-events.mjs';

test('contact events interpolate both threshold crossings and flag clipped episodes', () => {
  const events = contactEvents([0.2, 0, 0, 0.2, 0], 10, 0.1);
  assert.equal(events.length, 2);
  assert.equal(events[0].start, 0.05);
  assert.equal(events[0].end, 0.25);
  assert.equal(events[0].clippedStart, false);
  assert.equal(events[0].clippedEnd, false);
  assert.equal(events[1].clippedEnd, true);
  assert.equal(contactEvents([0, 0.2], 10, 0.1)[0].clippedStart, true);
  assert.deepEqual(contactEvents([0.2, 0.3], 10, 0.1), []);
  assert.throws(() => contactEvents([NaN]), /Nonfinite/);
});

test('early closures can be matched without requiring overlap with the phoneme', () => {
  const phone = { phone: 'P', start: 1, end: 1.02 };
  const event = { start: 0.9, end: 0.95, deepest: 0.92, clippedStart: false, clippedEnd: false };
  const result = associateContacts([phone], [event], 0.2)[0];
  assert.equal(result.status, 'matched');
  const offsets = offsetMeasurements(phone, event);
  assert.ok(Math.abs(offsets.onsetFromStartMs + 100) < 1e-8);
  assert.ok(Math.abs(offsets.releaseFromEndMs + 70) < 1e-8);
  assert.ok(Math.abs(offsets.midpointFromPhoneMidpointMs + 85) < 1e-8);
});

test('shared, ambiguous, absent, and boundary contacts do not enter primary statistics', () => {
  const event = { start: 0.9, end: 1, deepest: 0.95, clippedStart: false, clippedEnd: false };
  const phones = [{ phone: 'M', start: 1, end: 1.02 }, { phone: 'P', start: 1.1, end: 1.12 }];
  assert.ok(associateContacts(phones, [event]).every(({ status, event }) => status === 'shared_or_neighbor_ambiguous' && event === null));
  assert.equal(associateContacts([phones[0]], [event, { ...event, start: 1.05, end: 1.1 }])[0].status, 'multiple_events');
  assert.equal(associateContacts([phones[0]], [])[0].status, 'no_contact');
  assert.equal(associateContacts([phones[0]], [{ ...event, clippedStart: true }])[0].status, 'clipped_event');
  assert.deepEqual(associateContacts([{ phone: 'AA', start: 1, end: 1.1 }], [event]), []);
});

test('distribution reports central ranges rather than treating outliers as typical bounds', () => {
  const stats = distribution([0, 10, 20, 30, 1000]);
  assert.equal(stats.n, 5);
  assert.equal(stats.median, 20);
  assert.equal(stats.p05, 2);
  assert.ok(stats.p95 < stats.max);
  assert.deepEqual(distribution([]), { n: 0 });
});

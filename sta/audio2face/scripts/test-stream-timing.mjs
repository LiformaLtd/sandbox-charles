import { test } from 'node:test';
import assert from 'node:assert/strict';
import { observeFirstAudio } from '../src/pcm.js';

test('first TTS sample is observed before STA starts consuming, without draining audio', async () => {
  let release, notified, reads = 0, calls = 0, closed = false;
  const available = new Promise(resolve => { release = resolve; });
  const notification = new Promise(resolve => { notified = resolve; });
  async function* source() {
    try {
      reads++;
      yield await available;
      reads++;
      yield new Float32Array([2]);
    } finally { closed = true; }
  }
  const stream = observeFirstAudio(source(), () => { calls++; notified(); });
  release(new Float32Array([1]));
  await notification;
  assert.equal(reads, 1);
  const iterator = stream[Symbol.asyncIterator]();
  assert.deepEqual((await iterator.next()).value, new Float32Array([1]));
  assert.deepEqual((await iterator.next()).value, new Float32Array([2]));
  assert.equal(calls, 1);
  await iterator.return();
  assert.equal(closed, true);
});

test('producer failure is still delivered to the animation consumer', async () => {
  const failure = new Error('TTS failed');
  async function* source() { throw failure; }
  const stream = observeFirstAudio(source(), () => assert.fail('No audio produced'));
  await assert.rejects(stream[Symbol.asyncIterator]().next(), failure);
});

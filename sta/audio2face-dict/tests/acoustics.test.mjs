import { test } from 'node:test';
import assert from 'node:assert/strict';
import { describeSpeech, acousticDistance } from '../src/lib/dictionary/acoustics.mjs';
import { closureWarps, warpSourceTime } from '../src/lib/dictionary/retime-closures.mjs';
import { buildIndex, CHANNELS } from '../src/lib/dictionary/index.mjs';
import { planAnimation } from '../src/lib/dictionary/animate.mjs';

test('acoustic descriptors tolerate recording gain and silent boundary windows', () => {
  const pcm = Float32Array.from({ length: 16000 }, (_, i) => 0.2 * Math.sin(i * 2 * Math.PI * 120 / 16000));
  const phones = [{ phone: 'AA', start: 0.1, end: 0.3 }, { phone: 'AA', start: 0.5, end: 0.8 }];
  const quiet = describeSpeech(pcm.map((value) => value / 4), phones);
  const loud = describeSpeech(pcm, phones);
  for (let i = 0; i < phones.length; i++) assert.ok(acousticDistance(quiet[i], loud[i]) < 0.001);
  const silence = describeSpeech(new Float32Array(320), [{ phone: 'P', start: 0, end: 0.02 }]);
  assert.ok(silence.flat().every(Number.isFinite));
  assert.deepEqual(describeSpeech(pcm, []), []);
});

test('acoustic matching chooses the better sounding source with identical phonemes and timing', () => {
  const phones = [{ phone: 'AA', start: 0.1, end: 0.2 }, { phone: 'IY', start: 0.2, end: 0.3 }];
  const pcm = new Float32Array(8000);
  const acoustic = describeSpeech(pcm, phones);
  const clips = [0, 1].map((i) => ({ id: 'clip-' + i, phones, frameCount: 30, frameOffset: i * 30,
    runs: [{ from: 0, to: 2 }], acoustics: acoustic.map((row) => row.map((value, feature) => value + (i === 0 && feature < 6 ? 3 : 0))) }));
  const dictionary = { clips, index: buildIndex(clips), weights: new Float32Array(60 * CHANNELS) };
  const selected = (options) => planAnimation(dictionary, phones, pcm, options).segments.find(({ kind }) => kind === 'pair').pieces[0].match.clip.id;
  assert.equal(selected({ acousticAware: false }), 'clip-0');
  assert.equal(selected({ acousticAware: true, acousticMode: 'similarity' }), 'clip-1');
  assert.equal(selected({}), 'clip-0');
  assert.equal(selected({ acousticAware: true }), 'clip-0');
});

test('closure retiming is monotone, bounded, and preserves support endpoints', () => {
  const source = [{ phone: 'AA', start: 0, end: 0.1 }, { phone: 'P', start: 0.14, end: 0.18 }, { phone: 'AA', start: 0.22, end: 0.32 }];
  const guide = { samples: new Map([[source[1], [{ time: 0.16, confidence: 0.95 }]]]), gapAt: (_, time) => Math.abs(time - 0.15) < 0.005 ? 0.03 : 0.5 };
  const warps = closureWarps(guide, {}, source, source);
  assert.equal(warps.length, 1);
  const { start, anchor, contact, end } = warps[0];
  assert.equal(warpSourceTime(warps, start), start);
  assert.equal(warpSourceTime(warps, end), end);
  assert.equal(warpSourceTime(warps, anchor), contact);
  let previous = -Infinity;
  for (let i = 0; i <= 100; i++) {
    const time = i / 100 * 0.32;
    const mapped = warpSourceTime(warps, time);
    assert.ok(mapped >= previous);
    assert.ok(Math.abs(mapped - time) <= 0.04);
    previous = mapped;
  }
  guide.samples.get(source[1])[0].confidence = 0.5;
  assert.equal(closureWarps(guide, {}, source, source), undefined);
  guide.samples.get(source[1])[0].confidence = 0.95;
  guide.gapAt = () => 0.5;
  assert.equal(closureWarps(guide, {}, source, source), undefined);
});

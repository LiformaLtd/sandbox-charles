import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { advanceClosureFeatures, closurePrediction } from '../src/lib/dictionary/closures.mjs';

const root = new URL('../', import.meta.url);
const profile = JSON.parse(await readFile(new URL('data/closure-windows.json', root)));
const examples = [], clips = new Set();
const trainingHash = createHash('sha256');
for (const name of ['full', 'targeted', 'subtitles']) {
  const directory = new URL('.local/' + name + '/', root);
  const rows = JSON.parse(await readFile(new URL('clips.json', directory)));
  for (const row of rows.filter(({ split }) => split === 'train')) {
    assert.ok(!clips.has(row.id), 'Duplicate training clip'); clips.add(row.id);
    trainingHash.update(JSON.stringify({ id: row.id, samples: row.samples, phones: row.phones }));
    const readFloats = async (extension) => { const b = await readFile(new URL(row.id + extension, directory)); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
    const [pcm, gaps] = await Promise.all(['.pcm','.geometry-gaps'].map(readFloats));
    assert.equal(gaps.length, Math.ceil(pcm.length * 60 / 16000) * 9);
    for (const phone of row.phones) {
      const band = profile.phones[phone.phone];
      if (!band) continue;
      const start = phone.start + band.onsetRangeMs[0] / 1000 - .02;
      const end = phone.end + band.releaseRangeMs[1] / 1000 + .02;
      for (let frame = Math.max(0, Math.ceil(start * 60)); frame < Math.min(gaps.length / 9, Math.ceil(end * 60)); frame++) {
        examples.push({ features: advanceClosureFeatures(pcm, phone, frame / 60), contact: Number(gaps[frame * 9 + 4] <= profile.threshold) });
      }
    }
  }
}
assert.equal(clips.size, profile.trainingClips, 'Training corpus differs from timing profile');
assert.equal(trainingHash.digest('hex'), profile.trainingSetSha256, 'Training manifest/timings differ from timing profile');
const width = examples[0].features.length;
const mean = Array.from({ length: width }, (_, i) => examples.reduce((sum, row) => sum + row.features[i], 0) / examples.length);
const scale = mean.map((value, i) => Math.max(.001, Math.sqrt(examples.reduce((sum, row) => sum + (row.features[i] - value) ** 2, 0) / examples.length)));
const x = examples.map(({ features }) => features.map((value, i) => (value - mean[i]) / scale[i]));
const model = { version: 1, mean, scale, weights: Array(width).fill(0), bias: 0,
  trainingClips: clips.size, trainingFrames: examples.length, trainingSetSha256: profile.trainingSetSha256, timingProfile: profile,
  featureTiming: 'Offset from phoneme start divided by 100 ms; squared offset; duration and acoustic/context features', regularization: .001, iterations: 900 };
for (let iteration = 0; iteration < 900; iteration++) {
  const gradient = Array(width).fill(0); let bias = 0;
  for (let n = 0; n < x.length; n++) {
    let value = model.bias;
    for (let i = 0; i < width; i++) value += model.weights[i] * x[n][i];
    const error = 1 / (1 + Math.exp(-value)) - examples[n].contact;
    bias += error;
    for (let i = 0; i < width; i++) gradient[i] += error * x[n][i];
  }
  model.bias -= .15 * bias / x.length;
  for (let i = 0; i < width; i++) model.weights[i] -= .15 * (gradient[i] / x.length + .001 * model.weights[i]);
}
await writeFile(new URL('data/advance-closure-model.json', root), JSON.stringify(model, null, 2) + '\n');
const tally = { tp: 0, fp: 0, fn: 0 };
for (const row of examples) {
  const predicted = closurePrediction(model, row.features) >= .85;
  if (predicted && row.contact) tally.tp++;
  else if (predicted) tally.fp++;
  else if (row.contact) tally.fn++;
}
console.log(JSON.stringify({ trainingClips: clips.size, frames: examples.length, trainingOnlyAt085: tally, precision: tally.tp / (tally.tp + tally.fp) }));

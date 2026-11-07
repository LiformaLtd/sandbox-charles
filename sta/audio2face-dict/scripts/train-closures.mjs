import { readFile, writeFile } from 'node:fs/promises';
import { FPS, SAMPLE_RATE } from '../src/lib/dictionary/index.mjs';
import { closureFeatures, closurePhone, closurePrediction, CONTACT_GAP } from '../src/lib/dictionary/closures.mjs';

const directories = ['full', 'targeted'].map((name) => new URL('../.local/' + name + '/', import.meta.url));
const examples = [];
for (const directory of directories) {
  const rows = JSON.parse(await readFile(new URL('clips.json', directory)));
  for (const row of rows.filter(({ split }) => split === 'train')) {
    if (!row.phones.some(({ phone }) => closurePhone(phone))) continue;
    const [pcmBytes, gapsBytes] = await Promise.all(['pcm', 'geometry-gaps'].map((ext) =>
      readFile(new URL(row.id + '.' + ext, directory))));
    const pcm = new Float32Array(pcmBytes.buffer.slice(pcmBytes.byteOffset, pcmBytes.byteOffset + pcmBytes.byteLength));
    const gaps = new Float32Array(gapsBytes.buffer.slice(gapsBytes.byteOffset, gapsBytes.byteOffset + gapsBytes.byteLength));
    if (gaps.length !== Math.ceil(pcm.length * FPS / SAMPLE_RATE) * 9) throw new Error('Invalid lip gap track for ' + row.id);
    for (const phone of row.phones.filter(({ phone }) => closurePhone(phone))) {
      for (let frame = Math.max(0, Math.ceil((phone.start - 0.02) * FPS));
        frame < Math.min(gaps.length / 9, Math.ceil((phone.end + 0.02) * FPS)); frame++) {
        examples.push({
          features: closureFeatures(pcm, phone, frame / FPS),
          contact: Number(gaps[frame * 9 + 4] <= CONTACT_GAP),
        });
      }
    }
  }
}
if (!examples.length) throw new Error('No closure training examples');
const width = examples[0].features.length;
const mean = Array.from({ length: width }, (_, i) =>
  examples.reduce((sum, { features }) => sum + features[i], 0) / examples.length);
const scale = mean.map((value, i) => Math.max(0.001, Math.sqrt(
  examples.reduce((sum, { features }) => sum + (features[i] - value) ** 2, 0) / examples.length)));
const model = { version: 1, mean, scale, weights: Array(width).fill(0), bias: 0 };
for (let iteration = 0; iteration < 600; iteration++) {
  const gradient = Array(width).fill(0);
  let bias = 0;
  for (const { features, contact } of examples) {
    const error = closurePrediction(model, features) - contact;
    bias += error;
    for (let i = 0; i < width; i++) gradient[i] += error * (features[i] - mean[i]) / scale[i];
  }
  model.bias -= 0.15 * bias / examples.length;
  for (let i = 0; i < width; i++) model.weights[i] -= 0.15 * (gradient[i] / examples.length + 0.01 * model.weights[i]);
}
await writeFile(new URL('../.local/closure-model.json', import.meta.url), JSON.stringify(model));
console.log(`Trained on ${examples.length} frames across the two training corpora.`);

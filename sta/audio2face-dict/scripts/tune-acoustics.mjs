import { readFile, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { planAnimation, bakeAnimation } from '../src/lib/dictionary/animate.mjs';
const json = async (file) => JSON.parse(await readFile(file));
const floats = async (file) => { const b = await readFile(file); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const dictionary = await json('static/dictionary/index.json');
assert.ok(dictionary.selectionModel && dictionary.clips.every((clip) => clip.acoustics), 'Prepare the selection model and pack with --acoustics before tuning');
// This sweep is restricted to development clips. Fixed fresh/test sets are
// evaluated separately for acceptance, never included in the ridge fit.
dictionary.weights = await floats('static/dictionary/' + dictionary.weightsFile);
const rows = (await json('.local/full/clips.json')).filter((row) => row.split === 'development');
const data = await Promise.all(rows.map(async (row) => ({ ...row, pcm: await floats('.local/full/' + row.id + '.pcm'), geometry: await floats('.local/full/' + row.id + '.geometry-gaps') })));
const configs = [{ name: 'baseline', acousticAware: false, closureRetiming: false }, { name: 'retiming', acousticAware: false, closureRetiming: true }];
for (const weight of [0.1, 0.25, 0.5, 1]) for (const retime of [false, true]) configs.push({ name: `acoustic-${weight}-${retime ? 'retime' : 'linear'}`, acousticAware: true, acousticWeight: weight, closureRetiming: retime });
const reports = [];
for (const config of configs) {
  let squares = 0, steps = 0, count = 0, tp = 0, fp = 0, fn = 0, elapsed = 0;
  for (const row of data) {
    const started = performance.now();
    const plan = planAnimation(dictionary, row.phones, row.pcm, config);
    const frames = bakeAnimation(dictionary, plan);
    elapsed += performance.now() - started;
    const { neutral, deltas } = dictionary.lipRig;
    const predicted = new Float32Array(row.geometry.length);
    for (let f = 0; f < frames.length / 52; f++) for (let j = 0; j < 9; j++) {
      let gap = neutral[j];
      for (let c = 0; c < 52; c++) gap += deltas[c][j] * frames[f * 52 + c];
      predicted[f * 9 + j] = gap;
    }
    for (let i = 0; i < predicted.length; i++) {
      squares += (predicted[i] - row.geometry[i]) ** 2;
      if (i >= 9) steps += ((predicted[i] - predicted[i - 9]) - (row.geometry[i] - row.geometry[i - 9])) ** 2;
      count++;
    }
    for (const phone of row.phones.filter((p) => ['M', 'P', 'B'].includes(p.phone))) {
      let expected = false, actual = false;
      for (let f = Math.max(0, Math.ceil((phone.start - .02) * 60)); f < Math.min(frames.length / 52, Math.ceil((phone.end + .02) * 60)); f++) {
        expected ||= row.geometry[f * 9 + 4] <= .1;
        actual ||= predicted[f * 9 + 4] <= .1;
      }
      tp += expected && actual; fp += !expected && actual; fn += expected && !actual;
    }
  }
  reports.push({ ...config, sentences: rows.length, rms: Math.sqrt(squares / count), stepRms: Math.sqrt(steps / count), tp, fp, fn, meanMs: elapsed / rows.length });
}
await writeFile('.local/acoustic-development.json', JSON.stringify(reports, null, 2) + '\n');
console.log(JSON.stringify(reports, null, 2));

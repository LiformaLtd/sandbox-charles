import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { bakeAnimation, planAnimation } from '../src/lib/dictionary/animate.mjs';
import { CHANNELS, FPS } from '../src/lib/dictionary/index.mjs';

const packed = new URL('../static/dictionary/', import.meta.url);
const dictionary = JSON.parse(await readFile(new URL('index.json', packed)));
const local = new URL('../.local/' + (['targeted', 'subtitles'].includes(dictionary.corpus) ? 'full' : dictionary.corpus) + '/', import.meta.url);
const rows = JSON.parse(await readFile(new URL('clips.json', local))).filter(({ split }) => split === 'test');
const lipRig = JSON.parse(await readFile(new URL('lip-rig.json', local)));
const readFloats = async (url) => new Float32Array(await readFile(url).then((bytes) =>
  bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
dictionary.weights = await readFloats(new URL(dictionary.weightsFile, packed));
const nLips = lipRig.neutral.length;
function lipGap(track, frame, landmark) {
  let gap = lipRig.neutral[landmark];
  for (let i = 0; i < CHANNELS; i++) gap += track[frame * CHANNELS + i] * lipRig.deltas[i][landmark];
  return gap;
}

const coverage = { triple: 0, pair: 0, single: 0, fallback: 0, miss: 0 };
const joins = {};
const closure = { geometry: 0, fitted: 0, dictionary: 0, dictionaryAndFitted: 0, count: 0 };
let lipDifference = 0;
let fitDifference = 0;
let values = 0;
let lookup = 0;
let slowest = 0;
for (const clip of rows) {
  const [pcm, fitted, geometry] = await Promise.all(['pcm', dictionary.fit === 'lip' ? 'lip.arkit' : 'arkit', 'geometry-gaps'].map((extension) =>
    readFloats(new URL(clip.id + '.' + extension, local))));
  const start = performance.now();
  const plan = planAnimation(dictionary, clip.phones, pcm);
  const motion = bakeAnimation(dictionary, plan);
  const elapsed = performance.now() - start;
  lookup += elapsed;
  slowest = Math.max(elapsed, slowest);
  assert.equal(motion.length, fitted.length);
  assert.equal(geometry.length, motion.length / CHANNELS * nLips);
  assert.ok(motion.every(Number.isFinite));
  for (const key of Object.keys(coverage)) coverage[key] += plan.coverage[key];
  for (const join of plan.joins) {
    const key = [join.outgoing.match.count, join.incoming.match.count].sort().join('-');
    joins[key] = (joins[key] ?? 0) + 1;
  }
  for (const phone of clip.phones) {
    if (!['M', 'P', 'B'].includes(phone.phone)) continue;
    const from = Math.max(0, Math.ceil((phone.start - 0.02) * FPS));
    const to = Math.min(motion.length / CHANNELS, Math.ceil((phone.end + 0.02) * FPS));
    const center = Math.floor(nLips / 2);
    const minima = { geometry: Infinity, fitted: Infinity, dictionary: Infinity };
    for (let f = from; f < to; f++) {
      minima.geometry = Math.min(minima.geometry, geometry[f * nLips + center]);
      minima.fitted = Math.min(minima.fitted, lipGap(fitted, f, center));
      minima.dictionary = Math.min(minima.dictionary, lipGap(motion, f, center));
    }
    for (const track of Object.keys(minima)) if (minima[track] <= 0.1) closure[track]++;
    if (minima.dictionary <= 0.1 && minima.fitted <= 0.1) closure.dictionaryAndFitted++;
    closure.count++;
  }
  for (let f = 0; f < motion.length / CHANNELS; f++) {
    for (let i = 0; i < nLips; i++) {
      const baseline = lipGap(fitted, f, i);
      lipDifference += (lipGap(motion, f, i) - baseline) ** 2;
      fitDifference += (baseline - geometry[f * nLips + i]) ** 2;
      values++;
    }
  }
}
const report = {
  corpus: dictionary.corpus, testClips: rows.length, coverage, joins,
  centerLipContacts: closure, closureThreshold: 0.1,
  lipGapRms: { dictionaryVsFitted: Math.sqrt(lipDifference / values), fittedVsGeometry: Math.sqrt(fitDifference / values) },
  lookupMs: { mean: lookup / rows.length, max: slowest },
  dictionaryBytes: dictionary.weights.byteLength + Buffer.byteLength(await readFile(new URL('index.json', packed))),
};
console.log(JSON.stringify(report, null, 2));
await writeFile(new URL(['targeted', 'subtitles'].includes(dictionary.corpus) ? 'evaluation-' + dictionary.corpus + '.json' : 'evaluation.json', local), JSON.stringify(report, null, 2) + '\n');

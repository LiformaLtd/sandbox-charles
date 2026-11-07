import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { bakeAnimation, planAnimation } from '../src/lib/dictionary/animate.mjs';

const root = new URL('../.local/fresh/', import.meta.url);
const readJson = async (url) => JSON.parse(await readFile(url));
const packed = new URL('../static/dictionary/', import.meta.url);
const index = await readJson(new URL('index.json', packed));
const readFloats = async (url) => new Float32Array(await readFile(url).then((bytes) =>
  bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
index.weights = await readFloats(new URL(index.weightsFile, packed));
const corpusSources = index.corpus === 'targeted' ? ['full', 'targeted'] : [index.corpus];
const normalize = (text) => text.toLowerCase().replace(/[^a-z]+/g, ' ').trim();
const seen = new Set();
for (const source of corpusSources) {
  const rows = await readJson(new URL('../' + source + '/clips.json', root));
  for (const row of rows) seen.add(normalize(row.text));
}
const rows = await readJson(new URL('clips.json', root));
const groups = {};
const empty = () => ({ sentences: 0, triple: 0, pair: 0, single: 0, fallback: 0, miss: 0, isolatedMiss: 0, missSeconds: 0 });
const fallback = process.argv.includes('--fallback');
const misses = {};
for (const row of rows) {
  assert.ok(!seen.has(normalize(row.text)), 'Evaluation prompt was already in the corpus: ' + row.text);
  const pcm = await readFloats(new URL(row.id + '.pcm', root));
  const plan = planAnimation(index, row.phones, pcm, { singleFallback: fallback });
  assert.ok(bakeAnimation(index, plan).every(Number.isFinite), 'Nonfinite frames for ' + row.text);
  const gaps = plan.segments.filter((segment) => segment.kind === 'miss' && segment.first && segment.last);
  for (const { first, last } of gaps) {
    const key = first.phone + ' ' + last.phone;
    misses[key] = (misses[key] ?? 0) + 1;
  }
  for (const group of [row.split, 'total']) {
    const tally = groups[group] ??= empty();
    tally.sentences++;
    for (const [kind, count] of Object.entries(plan.coverage)) tally[kind] += count;
    tally.isolatedMiss += plan.coverage.miss - gaps.length;
    tally.miss -= plan.coverage.miss - gaps.length;
    tally.missSeconds += gaps.reduce((sum, { start, end }) => sum + end - start, 0);
  }
}
for (const tally of Object.values(groups)) {
  const total = tally.triple + tally.pair + tally.fallback + tally.miss;
  tally.exactPercent = total ? 100 * (tally.triple + tally.pair) / total : 0;
  tally.fallbackPercent = total ? 100 * tally.fallback / total : 0;
  tally.missSeconds = Number(tally.missSeconds.toFixed(3));
}
console.log(JSON.stringify({ corpus: index.corpus, fallback, groups, missingPairs: misses }, null, 2));

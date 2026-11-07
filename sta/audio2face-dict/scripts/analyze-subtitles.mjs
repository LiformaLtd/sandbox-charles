import { createReadStream } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';

const [input, maximum = '0'] = process.argv.slice(2);
if (!input) throw new Error('Usage: node scripts/analyze-subtitles.mjs /path/to/subtitles.txt [max-lines]');
const maxLines = Number(maximum);
if (!Number.isSafeInteger(maxLines) || maxLines < 0) throw new Error('max-lines must be a nonnegative integer');
const root = new URL('../.local/subtitle-analysis/', import.meta.url);
await mkdir(root, { recursive: true });
const dictionary = JSON.parse(await readFile(new URL('../static/models/lexicon.json', import.meta.url)));
const packed = JSON.parse(await readFile(new URL('../static/dictionary/index.json', import.meta.url)));
const labels = Object.keys(dictionary.phone2idx).filter((phone) => /^[A-Z]+$/.test(phone));
const phoneIds = new Map(labels.map((phone, i) => [phone, i]));
const width = labels.length;
const pronunciations = new Map(Object.entries(dictionary.lexicon).map(([word, phones]) =>
  [word.toLowerCase(), phones.map((phone) => phoneIds.get(phone))]));
const normalize = (text) => text.toLowerCase().replace(/[^a-z]+/g, ' ').trim();
const heldOut = new Set();
for (const name of ['full', 'fresh']) {
  const clips = JSON.parse(await readFile(new URL('../.local/' + name + '/clips.json', import.meta.url)));
  for (const clip of clips) if (clip.split !== 'train') heldOut.add(normalize(clip.text));
}
const counts = new Float64Array(width * width);
const withinWord = new Float64Array(counts.length);
const interWord = new Float64Array(counts.length);
const pools = Array.from({ length: counts.length }, () => []);
const poolSize = 12;
const stats = { lines: 0, words: 0, matchedWords: 0, unknownWords: 0, numericTokens: 0, candidateLines: 0, pairTokens: 0 };
const started = performance.now();
let progressAt = started;

const lines = createInterface({ input: createReadStream(resolve(input)), crlfDelay: Infinity });
for await (const original of lines) {
  stats.lines++;
  const text = original.replaceAll('’', "'").trim();
  const tokens = text.match(/[a-z]+(?:'[a-z]+)*|\d+/gi) ?? [];
  let previous = -1;
  let known = true;
  const pairSet = new Set();
  for (const token of tokens) {
    const phones = pronunciations.get(token.toLowerCase());
    if (!phones) {
      if (/^\d+$/.test(token)) stats.numericTokens++;
      else stats.unknownWords++;
      known = false;
      previous = -1;
      continue;
    }
    stats.matchedWords++;
    for (const [i, phone] of phones.entries()) {
      if (previous >= 0) {
        const pair = previous * width + phone;
        counts[pair]++;
        (i === 0 ? interWord : withinWord)[pair]++;
        pairSet.add(pair);
        stats.pairTokens++;
      }
      previous = phone;
    }
  }
  stats.words += tokens.length;
  if (known && tokens.length >= 6 && tokens.length <= 16 &&
      /^[A-Z][a-zA-Z\s.,!?'\-]*[.!?]$/.test(text) && text.length <= 180 &&
      !heldOut.has(normalize(text))) {
    stats.candidateLines++;
    const candidate = { text, pairs: [...pairSet], words: tokens.length, score: pairSet.size / tokens.length };
    for (const pair of pairSet) {
      const pool = pools[pair];
      if (pool.length === poolSize && pool.at(-1).score >= candidate.score) continue;
      if (pool.some((row) => row.text.toLowerCase() === text.toLowerCase())) continue;
      const at = pool.findIndex((row) => row.score < candidate.score);
      if (at < 0) pool.push(candidate);
      else pool.splice(at, 0, candidate);
      if (pool.length > poolSize) pool.pop();
    }
  }
  if (stats.lines % 1000000 === 0 && performance.now() - progressAt > 15000) {
    progressAt = performance.now();
    console.error(`${stats.lines.toLocaleString()} lines, ${stats.pairTokens.toLocaleString()} pairs, ${((progressAt - started) / 1000).toFixed(0)} s`);
  }
  if (maxLines && stats.lines >= maxLines) { lines.close(); break; }
}
const pairLabel = (id) => labels[Math.floor(id / width)] + ' ' + labels[id % width];
const current = new Set(Object.keys(packed.index).filter((key) => key.split(' ').length === 2).map((key) => {
  const [left, right] = key.split(' ');
  return phoneIds.get(left) * width + phoneIds.get(right);
}));
const candidates = [...new Set(pools.flat())];
const observed = [...counts.keys()].filter((id) => counts[id] > 0);
const mass = (covered) => observed.reduce((sum, id) => sum + (covered.has(id) ? counts[id] : 0), 0) / stats.pairTokens;

function choose(initial) {
  const covered = new Set(initial);
  const selected = [];
  const used = new Set();
  while (selected.length < 250) {
    let best, bestScore = 0;
    for (const candidate of candidates) {
      if (used.has(candidate.text)) continue;
      let gain = 0;
      for (const pair of candidate.pairs) {
        if (!covered.has(pair)) gain += Math.sqrt(counts[pair]);
      }
      const score = gain / candidate.words;
      if (score > bestScore) { best = candidate; bestScore = score; }
    }
    if (!best) break;
    used.add(best.text);
    const newPairs = best.pairs.filter((pair) => !covered.has(pair));
    for (const pair of best.pairs) covered.add(pair);
    selected.push({
      text: best.text, words: best.words, estimatedSeconds: best.words / 2.5,
      newPairs: newPairs.map(pairLabel), cumulativeUniquePairs: covered.size,
      cumulativeOccurrenceCoverage: mass(covered),
    });
  }
  return {
    sentences: selected.length, estimatedSeconds: selected.reduce((sum, row) => sum + row.estimatedSeconds, 0),
    uniquePairs: covered.size, occurrenceCoverage: mass(covered), selected,
    missingObservedPairs: observed.filter((id) => !covered.has(id)).map(pairLabel),
  };
}

const standalone = choose([]);
const augmentation = choose(current);
const frequencyRows = observed.map((id) => ({
  pair: pairLabel(id), occurrences: counts[id], withinWord: withinWord[id], interWord: interWord[id],
  inCurrentDictionary: current.has(id), candidates: pools[id].length,
})).sort((a, b) => b.occurrences - a.occurrences);
const summary = {
  input: resolve(input), stats, elapsedSeconds: (performance.now() - started) / 1000,
  phonemes: width, theoreticalPairs: width * width, observedPairs: observed.length,
  currentPairs: current.size, currentOccurrenceCoverage: mass(current),
  candidatePool: candidates.length,
  pronunciation: 'TinyAligner stress-free English CMUdict, first pronunciation, unknown words and numeric tokens break the pair sequence',
  timing: 'Text-derived candidates; synthesize and align before treating them as recorded animation coverage',
  standalone: { ...standalone, selected: undefined },
  augmentation: { ...augmentation, selected: undefined },
};
await Promise.all([
  writeFile(new URL('summary.json', root), JSON.stringify(summary, null, 2) + '\n'),
  writeFile(new URL('pair-frequencies.json', root), JSON.stringify(frequencyRows, null, 2) + '\n'),
  writeFile(new URL('standalone.json', root), JSON.stringify(standalone.selected, null, 2) + '\n'),
  writeFile(new URL('augmentation.json', root), JSON.stringify(augmentation.selected, null, 2) + '\n'),
]);
console.log(JSON.stringify(summary, null, 2));

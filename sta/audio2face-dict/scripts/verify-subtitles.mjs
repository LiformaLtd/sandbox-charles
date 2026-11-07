import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createSpeech, createAligner } from './speech.mjs';
import { speechRuns, validatePhones, key } from '../src/lib/dictionary/index.mjs';

const [selectionPath, maximum = '0'] = process.argv.slice(2);
if (!selectionPath) throw new Error('Usage: node scripts/verify-subtitles.mjs /path/to/augmentation.json [max-clips]');
const maxClips = Number(maximum);
if (!Number.isSafeInteger(maxClips) || maxClips < 0) throw new Error('max-clips must be a nonnegative integer');
const selected = JSON.parse(await readFile(resolve(selectionPath)));
const rows = maxClips ? selected.slice(0, maxClips) : selected;
const root = new URL('../.local/subtitle-analysis/', import.meta.url);
const audio = new URL('audio/', root);
await mkdir(audio, { recursive: true });
const packed = JSON.parse(await readFile(new URL('../static/dictionary/index.json', import.meta.url)));
const existing = new Set(Object.keys(packed.index).filter((pair) => pair.split(' ').length === 2));
const verified = new Set(existing);
const targetPairs = new Set();
const clips = [];
const aligner = await createAligner();
let speech;

async function save(name, value) {
  const temporary = new URL(name + '.partial', audio);
  await writeFile(temporary, value);
  await rename(temporary, new URL(name, audio));
}
try {
  for (const [i, row] of rows.entries()) {
    const id = createHash('sha256').update(row.text).digest('hex').slice(0, 16);
    let bytes;
    try {
      bytes = await readFile(new URL(id + '.pcm', audio));
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      speech ??= await createSpeech();
      const samples = await speech.speak(row.text);
      bytes = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      await save(id + '.pcm', bytes);
    }
    const pcm = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    let phones;
    try {
      phones = JSON.parse(await readFile(new URL(id + '.phones.json', audio)));
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      phones = await aligner.phones(pcm, row.text);
      await save(id + '.phones.json', JSON.stringify(phones));
    }
    validatePhones(phones, pcm.length / 16000);
    const actual = new Set();
    const allPairs = new Set(phones.slice(1).map((_, index) => key(phones, index, 2)));
    for (const run of speechRuns(phones, pcm, packed.silenceRms)) {
      for (let at = run.from; at + 1 < run.to; at++) actual.add(key(phones, at, 2));
    }
    const added = [...actual].filter((pair) => !verified.has(pair));
    for (const pair of actual) {
      verified.add(pair);
      targetPairs.add(pair);
    }
    clips.push({
      id, text: row.text, split: 'train', samples: pcm.length, phones,
      newPairs: added, confirmedPairs: [...actual],
      pauseExcludedPairs: [...allPairs].filter((pair) => !actual.has(pair)),
    });
    console.error(`${i + 1}/${rows.length} ${(pcm.length / 16000).toFixed(2)} s, ${added.length} new pairs: ${row.text}`);
  }
  const frequencies = JSON.parse(await readFile(new URL('pair-frequencies.json', root)));
  const total = frequencies.reduce((sum, row) => sum + row.occurrences, 0);
  const mass = (keys) => frequencies.reduce((sum, row) => sum + (keys.has(row.pair) ? row.occurrences : 0), 0) / total;
  const report = {
    sentences: clips.length, audioSeconds: clips.reduce((sum, row) => sum + row.samples / 16000, 0),
    originalPairs: existing.size, verifiedPairs: verified.size, newPairs: verified.size - existing.size,
    selectionUniquePairs: targetPairs.size,
    originalOccurrenceCoverage: mass(existing), verifiedOccurrenceCoverage: mass(verified),
    missingObservedPairs: frequencies.filter(({ pair }) => !verified.has(pair)),
  };
  await writeFile(new URL('verified-clips.json', root), JSON.stringify(clips, null, 2) + '\n');
  await writeFile(new URL('verified-summary.json', root), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ ...report, missingObservedPairs: report.missingObservedPairs.length }, null, 2));
} finally {
  speech?.close();
  await aligner.close();
}

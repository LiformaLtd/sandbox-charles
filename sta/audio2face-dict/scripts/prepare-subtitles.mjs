import assert from 'node:assert/strict';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';

const root = new URL('../.local/', import.meta.url);
const json = async (url) => JSON.parse(await readFile(url));
const source = new URL('subtitle-analysis/', root);
const target = new URL('subtitles/', root);
const rows = await json(new URL('verified-clips.json', source));
const selection = await json(new URL('augmentation.json', source));
assert.equal(rows.length, selection.length, 'Verify every selected subtitle before preparing the corpus');
const normalize = (text) => text.toLowerCase().replace(/[^a-z]+/g, ' ').trim();
const seen = new Set();
for (const name of ['full', 'targeted', 'fresh']) {
  for (const row of await json(new URL(name + '/clips.json', root))) seen.add(normalize(row.text));
}
const manifest = rows.map((row, i) => {
  assert.equal(row.text, selection[i].text, 'Verified selection order differs');
  assert.match(row.id, /^[0-9a-f]{16}$/);
  assert.equal(row.split, 'train');
  assert.ok(!seen.has(normalize(row.text)), 'Duplicate or evaluation transcript: ' + row.text);
  seen.add(normalize(row.text));
  return Object.fromEntries(['id', 'text', 'split', 'samples', 'phones'].map((key) => [key, row[key]]));
});
assert.equal(new Set(manifest.map(({ id }) => id)).size, manifest.length, 'Duplicate subtitle ids');
await mkdir(target, { recursive: true });
for (const row of manifest) {
  const audio = new URL('audio/' + row.id + '.pcm', source);
  assert.equal((await readFile(audio)).byteLength, row.samples * 4, 'Incomplete audio for ' + row.id);
  await copyFile(audio, new URL(row.id + '.pcm', target));
}
await writeFile(new URL('clips.json', target), JSON.stringify(manifest, null, 2) + '\n');
console.log(`Prepared ${manifest.length} subtitle training clips; no overlap with existing training or evaluation transcripts.`);

import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { buildIndex, CHANNELS, FPS, SAMPLE_RATE, speechRuns, validatePhones } from '../src/lib/dictionary/index.mjs';
import { validClosureModel } from '../src/lib/dictionary/closures.mjs';

const [mode = 'pilot', option, fit = 'lip', ...extra] = process.argv.slice(2);
if ((option && option !== '--fit') || (option && !process.argv[4]) || extra.length ||
    !['lip', 'baseline'].includes(fit)) throw new Error('Use --fit lip or --fit baseline');
const weightExtension = fit === 'lip' ? '.lip.arkit' : '.arkit';
if (!['pilot', 'full', 'targeted', 'subtitles'].includes(mode)) throw new Error('Usage: npm run pack -- pilot|full|targeted|subtitles');
const sources = (mode === 'subtitles' ? ['full', 'targeted', 'subtitles'] : mode === 'targeted' ? ['full', 'targeted'] : [mode])
  .map((name) => new URL('../.local/' + name + '/', import.meta.url));
const target = new URL('../static/dictionary/', import.meta.url);
const expectedChannels = JSON.parse(await readFile(new URL('../src/lib/studio/channels.json', import.meta.url)));
const rows = [];
for (const source of sources) {
  const channels = JSON.parse(await readFile(new URL('channels.json', source)));
  if (JSON.stringify(channels) !== JSON.stringify(expectedChannels)) throw new Error('ARKit channel order differs from the James rig');
  const manifest = JSON.parse(await readFile(new URL('clips.json', source)));
  rows.push(...manifest.filter((clip) => clip.split === 'train').map((clip) => ({ ...clip, source })));
}
const normalized = rows.map(({ text }) => text.toLowerCase().replace(/[^a-z]+/g, ' ').trim());
if (new Set(normalized).size !== normalized.length) throw new Error('Duplicate training transcripts in dictionary sources');
const silenceRms = 0.008;
const clips = [];
const tracks = [];
let frameOffset = 0;
for (const row of rows) {
  const pcm = await readFile(new URL(row.id + '.pcm', row.source));
  const weights = await readFile(new URL(row.id + weightExtension, row.source));
  const frameCount = Math.ceil(row.samples * FPS / SAMPLE_RATE);
  if (pcm.length !== row.samples * 4 || weights.length !== frameCount * CHANNELS * 4) {
    throw new Error(row.id + ': audio and animation durations do not match');
  }
  validatePhones(row.phones, row.samples / SAMPLE_RATE);
  const samples = new Float32Array(pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength));
  const runs = speechRuns(row.phones, samples, silenceRms).map(({ from, to }) => ({ from, to }));
  clips.push({ id: row.id, phones: row.phones, frameCount, frameOffset, runs });
  tracks.push(weights);
  frameOffset += frameCount;
}
const index = buildIndex(clips);
const binary = Buffer.concat(tracks);
const weightsFile = `weights-${createHash('sha256').update(binary).digest('hex').slice(0, 16)}.f32`;
const lipRig = fit === 'lip'
  ? JSON.parse(await readFile(new URL('lip-rig.json', sources[0])))
  : undefined;
let closureModel;
if (fit === 'lip' && ['targeted', 'subtitles'].includes(mode)) {
  try {
    closureModel = JSON.parse(await readFile(new URL('../.local/closure-model.json', import.meta.url)));
    if (!validClosureModel(closureModel)) throw new Error('Invalid closure model. Rerun scripts/train-closures.mjs.');
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
}
const bundle = { lipRig, closureModel, version: 1, corpus: mode, fit, channels: expectedChannels, fps: FPS, silenceRms, clips, index, frameCount: frameOffset, weightsFile };
await mkdir(target, { recursive: true });
for (const [name, data] of [[weightsFile, binary], ['index.json', JSON.stringify(bundle)]]) {
  await writeFile(new URL(name + '.partial', target), data);
  await rename(new URL(name + '.partial', target), new URL(name, target));
}
console.log(`${mode} (${fit} fit${closureModel ? ', closure model' : ''}): ${clips.length} clips, ${(rows.reduce((sum, row) => sum + row.samples, 0) / SAMPLE_RATE).toFixed(1)} seconds, ${Object.keys(index).length} keys`);

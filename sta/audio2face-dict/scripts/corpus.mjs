import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { pilot, development, test } from './prompts.mjs';
import { freshGeneral, freshFocus, targeted } from './coverage-prompts.mjs';
import { corpusSelector } from './select-prompts.mjs';
import { createAligner, createSpeech } from './speech.mjs';

const mode = process.argv[2] ?? 'pilot';
const promptSets = {
  pilot: { train: pilot, development, test },
  full: { train: pilot, development, test },
  targeted: { train: targeted },
  fresh: { general: freshGeneral, focus: freshFocus },
};
if (!Object.hasOwn(promptSets, mode)) throw new Error('Usage: npm run corpus -- pilot|full|targeted|fresh [seconds]');
const budget = Number(process.argv[3] ?? 600);
if (!Number.isFinite(budget) || budget <= 0) throw new Error('The corpus budget must be positive seconds.');
const root = new URL('../.local/' + mode + '/', import.meta.url);
await mkdir(root, { recursive: true });
const aligner = await createAligner();
let speech;

async function atomic(name, value) {
  const temporary = new URL(name + '.partial', root);
  await writeFile(temporary, value);
  await rename(temporary, new URL(name, root));
}

async function makeClip(text, split) {
  const id = createHash('sha256').update(text).digest('hex').slice(0, 16);
  if (mode === 'full') {
    for (const extension of ['.pcm', '.phones.json', '.arkit']) {
      const name = id + extension;
      try {
        await copyFile(new URL('../pilot/' + name, root), new URL(name, root));
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
      }
    }
  }
  let bytes;
  try {
    bytes = await readFile(new URL(id + '.pcm', root));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    speech ??= await createSpeech();
    const pcm = await speech.speak(text);
    bytes = Buffer.from(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    await atomic(id + '.pcm', bytes);
  }
  const pcm = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  let phones;
  try {
    phones = JSON.parse(await readFile(new URL(id + '.phones.json', root)));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    phones = await aligner.phones(pcm, text);
    await atomic(id + '.phones.json', JSON.stringify(phones));
  }
  return { id, text, split, samples: pcm.length, phones };
}

try {
  const clips = [];
  for (const [split, prompts] of Object.entries(promptSets[mode])) {
    for (const text of prompts) {
      const clip = await makeClip(text, split);
      clips.push(clip);
      console.log(split, (clip.samples / 16000).toFixed(2), text);
    }
  }
  if (mode === 'full') {
    const next = corpusSelector(aligner.dictionary);
    let samples = clips.filter(({ split }) => split === 'train').reduce((sum, clip) => sum + clip.samples, 0);
    while (samples < budget * 16000) {
      const clip = await makeClip(next(), 'train');
      clips.push(clip);
      samples += clip.samples;
      console.log('train', (samples / 16000).toFixed(1) + '/' + budget + ' s', clip.text);
    }
  }
  await atomic('clips.json', JSON.stringify(clips, null, 2));
} finally {
  speech?.close();
  await aligner.close();
}

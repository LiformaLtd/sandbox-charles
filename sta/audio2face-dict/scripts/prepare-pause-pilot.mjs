import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { SAMPLE_RATE } from '../src/lib/dictionary/index.mjs';

// Reuse held-out speech, retaining 60ms of release and 120ms of preparation.
// Native references must be inferred on each complete new waveform, never
// assembled from the two individual reference tracks.
for (const [split, corpus] of [['development','full'],['fresh','fresh'],['test','full']]) {
  const source = new URL(`../.local/${corpus}/`, import.meta.url);
  const output = new URL(`../.local/pause-pilot/${split}/`, import.meta.url);
  await mkdir(output, { recursive: true });
  const rows = JSON.parse(await readFile(new URL('clips.json', source))).filter(r => corpus === 'fresh' || r.split === split);
  const clips = [];
  for (let i = 0; i + 1 < rows.length; i += 2) {
    const [a,b] = rows.slice(i,i+2);
    const [ab,bb] = await Promise.all([a,b].map(r => readFile(new URL(r.id + '.pcm', source))));
    const left = new Float32Array(ab.buffer.slice(ab.byteOffset, ab.byteOffset + ab.byteLength));
    const right = new Float32Array(bb.buffer.slice(bb.byteOffset, bb.byteOffset + bb.byteLength));
    const cut = Math.min(left.length, Math.ceil((a.phones.at(-1).end + .06) * SAMPLE_RATE));
    const from = Math.max(0, Math.floor((b.phones[0].start - .12) * SAMPLE_RATE));
    const desired = [.2,.35,.5][(i / 2) % 3];
    const silence = Math.max(0, Math.round(desired * SAMPLE_RATE - (cut - a.phones.at(-1).end * SAMPLE_RATE) - (b.phones[0].start * SAMPLE_RATE - from)));
    const pcm = new Float32Array(cut + silence + right.length - from);
    pcm.set(left.subarray(0,cut)); pcm.set(right.subarray(from), cut + silence);
    const offset = (cut + silence - from) / SAMPLE_RATE;
    const text = a.text + ' ' + b.text;
    const id = createHash('sha256').update(Buffer.from(pcm.buffer)).digest('hex').slice(0,16);
    const phones = [...a.phones, ...b.phones.map(p => ({ ...p, start: p.start + offset, end: p.end + offset }))];
    const row = { id, text, split, samples: pcm.length, phones, sourceClips: [a.id,b.id], intendedGapSeconds: desired };
    await writeFile(new URL(id + '.pcm', output), Buffer.from(pcm.buffer));
    await writeFile(new URL(id + '.phones.json', output), JSON.stringify(phones));
    clips.push(row);
  }
  await writeFile(new URL('clips.json', output), JSON.stringify(clips,null,2));
  console.log(split, clips.length, 'paired recordings');
}

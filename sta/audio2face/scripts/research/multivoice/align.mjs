// Offline alignment only. Uses frozen optimized TinyAligner feature extraction/CTC.
import { readFile, writeFile, rename, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve, join } from 'node:path';
import * as ort from 'onnxruntime-web/wasm';
const app = fileURLToPath(new URL('../../../', import.meta.url));
const root = resolve(process.env.A2F_MULTIVOICE_OUTPUT || join(app, '.local/compact-multivoice'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const planned = JSON.parse(await readFile(join(root, 'planned.json')));
const assets = JSON.parse(await readFile(join(root, 'assets.json'))).assets;
for (const [path, record] of Object.entries(assets)) {
  if (hash(await readFile(path)) !== record.sha256) throw Error(`Asset changed: ${path}`);
}
const { extractLogMel } = await import(pathToFileURL(join(root, 'sources/aligner/audio.ts')));
const { ctcForcedAlign, extractSegments } = await import(pathToFileURL(join(root, 'sources/aligner/ctc.ts')));
const dictionary = JSON.parse(await readFile(join(root, 'assets/aligner/lexicon.json')));
const phoneNames = Object.keys(dictionary.phone2idx).filter(p => /^[A-Z]+$/.test(p)).map(p => p.toLowerCase()).sort();
await writeFile(join(root, 'phone-classes.json'), JSON.stringify(phoneNames));
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(new Uint8Array(await readFile(join(root, 'assets/aligner/model.onnx'))), { executionProviders: ['wasm'] });
const manifest = [];
await mkdir(join(root, 'alignment'), { recursive: true });
try {
  for (const [i, plannedRow] of planned.entries()) {
    const row = JSON.parse(await readFile(join(root, 'synthesis', plannedRow.stem + '.json')));
    const data = await readFile(row.audio);
    if (hash(data) !== row.sha256_audio) throw Error(`Audio changed: ${row.stem}`);
    const recipe = hash(JSON.stringify({ audio: row.sha256_audio, assets,
      script: hash(await readFile(fileURLToPath(import.meta.url))) }));
    const path = join(root, 'alignment', row.stem + '.json');
    let record;
    try {
      record = JSON.parse(await readFile(path));
      if (record.alignment_recipe_sha256 !== recipe) throw Error(`Alignment changed: ${row.stem}`);
    } catch (e) {
      if (e.code !== 'ENOENT') throw e;
      const audio = new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
      const words = row.text.replaceAll('’', "'").match(/[a-z]+(?:'[a-z]+)*/gi);
      const phones = ['sil', ...words.flatMap(w => {
        if (!dictionary.lexicon[w.toUpperCase()]) throw Error(`Unknown word: ${w}`);
        return dictionary.lexicon[w.toUpperCase()];
      }), 'sil'];
      const ids = phones.map(p => dictionary.phone2idx[p]);
      if (ids.some(i => i === undefined)) throw Error('Missing phone ID');
      const features = extractLogMel(audio);
      const mel = new ort.Tensor('float32', features.mel, [1, features.frames, features.nMels]);
      const start = performance.now();
      let log_probs;
      try {
        ({ log_probs } = await session.run({ mel }));
        const [frames, vocabulary] = log_probs.dims;
        const required = ids.length + ids.slice(1).filter((id, index) => id === ids[index]).length;
        if (frames < required) throw Error(`Too few frames: ${row.stem}`);
        const ctcPath = ctcForcedAlign(log_probs.data, frames, vocabulary, ids, dictionary.phone2idx['<blank>']);
        const segments = extractSegments(ctcPath, phones, .02).filter(s => s.word);
        if (segments.length !== phones.length) throw Error(`Incomplete alignment: ${row.stem}`);
        const aligned = segments.slice(1, -1).map(s => ({ phone: s.word.toLowerCase(), start: s.start, end: Math.min(row.seconds, s.end) }));
        if (aligned.some((s, j) => s.end <= s.start || s.start < 0 || (j && s.start < aligned[j - 1].end - 1e-6))) {
          throw Error(`Invalid timing: ${row.stem}`);
        }
        record = { ...row, phones: aligned, alignment_recipe_sha256: recipe, alignment_ms: performance.now() - start };
        await writeFile(path + '.part', JSON.stringify(record, null, 2));
        await rename(path + '.part', path);
      } finally { mel.dispose(); log_probs?.dispose(); }
    }
    manifest.push(record);
    if (i % 16 === 0) console.log('Aligned', i + 1, '/', planned.length);
  }
  await writeFile(join(root, 'manifest.json.part'), JSON.stringify(manifest, null, 2));
  await rename(join(root, 'manifest.json.part'), join(root, 'manifest.json'));
} finally { await session.release(); }

// Offline corpus synthesis using the same pinned JS engines/resampler as the demo.
import { readFile, writeFile, mkdir, rename, open, unlink } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve, join } from 'node:path';
import init, { Model } from 'phonon-tts/raw';
import * as ort from 'onnxruntime-web/wasm';

const app = fileURLToPath(new URL('../../../', import.meta.url));
const root = resolve(process.env.A2F_MULTIVOICE_OUTPUT || join(app, '.local/compact-multivoice'));
const engine = process.argv[2];
if (!['phonon', 'supertonic'].includes(engine)) throw Error('Usage: synthesize.mjs phonon|supertonic [limit]');
const limit = Number(process.argv[3] || Infinity);
const hash = data => createHash('sha256').update(data).digest('hex');
const config = JSON.parse(await readFile(join(root, 'assets.json')));
const planned = JSON.parse(await readFile(join(root, 'planned.json'))).filter(r => r.engine === engine);
for (const [path, record] of Object.entries(config.assets)) {
  if (hash(await readFile(path)) !== record.sha256) throw Error(`Asset changed: ${path}`);
}
await mkdir(join(root, 'audio'), { recursive: true });
await mkdir(join(root, 'synthesis'), { recursive: true });
const lockPath = join(root, `${engine}.lock`);
const lock = await open(lockPath, 'wx');
await lock.writeFile(String(process.pid));
const { to16k } = await import(pathToFileURL(join(root, 'sources/pcm.js')));
async function resample(pcm, rate) {
  const chunks = [];
  for await (const chunk of to16k((async function* () { yield pcm; })(), rate)) chunks.push(chunk);
  const result = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
  return result;
}
async function atomic(path, bytes) {
  await writeFile(path + '.part', bytes);
  await rename(path + '.part', path);
}
function random(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = Math.imul(state ^ state >>> 15, 1 | state);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
let model;
const sessions = [];
const voiceIndices = new Map();
const styles = new Map();
let synth;
try {
  const missing = [];
  for (const row of planned) {
    const path = join(root, 'synthesis', row.stem + '.json');
    const recipe = hash(JSON.stringify({ row, assets: config.assets,
      implementation: hash(await readFile(fileURLToPath(import.meta.url))) }));
    try {
      const previous = JSON.parse(await readFile(path));
      if (previous.recipe_sha256 !== recipe || hash(await readFile(previous.audio)) !== previous.sha256_audio) {
        throw Error(`Cached synthesis changed: ${row.stem}. Use a fresh study directory.`);
      }
    } catch (e) { if (e.code !== 'ENOENT') throw e; missing.push({ ...row, recipe_sha256: recipe }); }
  }
  if (!missing.length) { console.log(engine, 'already complete'); }
  else if (engine === 'phonon') {
    await init({ module_or_path: await readFile(fileURLToPath(import.meta.resolve('phonon-tts/wasm/phonon_tts_simd_bg.wasm'))) });
    model = await Model.load(...await Promise.all(['model.q8.gguf', 'tokenizer.json', 'config.json'].map(k => readFile(config.phonon[k]))),
      'q8', 'en', undefined, undefined, 'cpu');
    for (const v of config.voices.filter(v => v.engine === engine)) voiceIndices.set(v.name, model.add_voice(await readFile(v.asset)));
    synth = async row => {
      const seed = createHash('sha256').update(row.stem + ':' + row.text).digest().readBigUInt64LE();
      model.start_generation(voiceIndices.get(row.voice), row.text, row.temperature, seed);
      try {
        const chunks = [];
        while (model.next_chunk() !== undefined) {
          for (let pcm; (pcm = await model.generation_step()) !== undefined;) chunks.push(pcm);
        }
        const result = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
        let offset = 0;
        for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
        return resample(result, model.sample_rate());
      } finally { model.stop_generation(); }
    };
  } else {
    ort.env.wasm.numThreads = 1;
    const { TextToSpeech, UnicodeProcessor, Style } = await import(pathToFileURL(join(root, 'sources/supertonic.js')));
    const cfg = JSON.parse(await readFile(join(config.supertonic_root, 'onnx/tts.json')));
    const index = JSON.parse(await readFile(join(config.supertonic_root, 'onnx/unicode_indexer.json')));
    for (const name of ['duration_predictor', 'text_encoder', 'vector_estimator', 'vocoder']) {
      console.log('Loading', name);
      sessions.push(await ort.InferenceSession.create(new Uint8Array(await readFile(join(config.supertonic_root, 'onnx', name + '.onnx'))),
        { executionProviders: ['wasm'] }));
    }
    const tts = new TextToSpeech(cfg, new UnicodeProcessor(index), ...sessions);
    for (const v of config.voices.filter(v => v.engine === engine)) {
      const data = JSON.parse(await readFile(v.asset));
      const tensor = key => new ort.Tensor('float32', Float32Array.from(data[key].data.flat(Infinity)), data[key].dims);
      styles.set(v.name, new Style(tensor('style_ttl'), tensor('style_dp')));
    }
    synth = async row => {
      const oldRandom = Math.random;
      Math.random = random(createHash('sha256').update(row.stem + ':' + row.text).digest().readUInt32LE());
      try {
        const { wav, duration } = await tts.batch([row.text], ['en'], styles.get(row.voice), row.supertonic_steps, row.supertonic_speed);
        return resample(Float32Array.from(wav.slice(0, Math.floor(duration[0] * tts.sampleRate))), tts.sampleRate);
      } finally { Math.random = oldRandom; }
    };
  }
  const started = performance.now();
  for (const [i, row] of missing.slice(0, limit).entries()) {
    const pcm = await synth(row);
    if (pcm.length < 1600 || [...pcm].some(v => !Number.isFinite(v))) throw Error(`Invalid PCM: ${row.stem}`);
    const audio = join(root, 'audio', row.stem + '.f32');
    const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    const record = { ...row, audio, samples: pcm.length, seconds: pcm.length / 16000,
      sha256_audio: hash(bytes), sample_rate: 16000 };
    await atomic(audio, bytes);
    await atomic(join(root, 'synthesis', row.stem + '.json'), JSON.stringify(record, null, 2));
    console.log(engine, i + 1, '/', Math.min(missing.length, limit), row.stem, record.seconds.toFixed(2), 'seconds audio;',
      ((performance.now() - started) / 1000).toFixed(1), 'seconds elapsed');
  }
} finally {
  model?.free();
  for (const style of styles.values()) { style.ttl.dispose(); style.dp.dispose(); }
  for (const session of sessions) await session.release();
  await lock.close();
  await unlink(lockPath);
}

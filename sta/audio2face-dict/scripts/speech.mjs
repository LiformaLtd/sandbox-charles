import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import init, { Model } from 'phonon-tts/raw';
import { concatPcm } from 'phonon-tts';
import * as ort from 'onnxruntime-web/wasm';
import { prepareTranscript, align } from '../src/lib/alignment/engine.ts';
import { to16k } from '../src/lib/studio/pcm.js';

export const readAsset = (path) => readFile(new URL('../static/' + path, import.meta.url));

export async function createSpeech() {
  const [wasm, weights, tokenizer, config, voice] = await Promise.all([
    readFile(new URL(import.meta.resolve('phonon-tts/wasm/phonon_tts_simd_bg.wasm'))),
    ...['model.q8.gguf', 'tokenizer.json', 'config.json', 'easton.safetensors']
      .map((name) => readAsset('assets/phonon/' + name)),
  ]);
  await init({ module_or_path: wasm });
  const model = await Model.load(weights, tokenizer, config, 'q8', 'en', undefined, undefined, 'cpu');
  const voiceIndex = model.add_voice(voice);
  async function* samples() {
    while (model.next_chunk() !== undefined) {
      for (let pcm; (pcm = await model.generation_step()) !== undefined;) yield pcm;
    }
  }
  return {
    async speak(text) {
      const seed = createHash('sha256').update(text).digest().readBigUInt64LE();
      model.start_generation(voiceIndex, text, 0.7, seed);
      try {
        const chunks = [];
        for await (const pcm of to16k(samples(), model.sample_rate())) chunks.push(pcm);
        return concatPcm(chunks);
      } finally {
        model.stop_generation();
      }
    },
    close() { model.free(); },
  };
}

export async function createAligner() {
  const dictionary = JSON.parse(await readAsset('models/lexicon.json'));
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(await readAsset('models/model.onnx'), {
    executionProviders: ['wasm'],
  });
  return {
    dictionary,
    async phones(pcm, text) {
      const words = await align(session, pcm, 16000, prepareTranscript(text, dictionary));
      return words.flatMap((word) => word.phones.map(({ word: phone, start, end }) => ({ phone, start, end })));
    },
    close() { return session.release(); },
  };
}

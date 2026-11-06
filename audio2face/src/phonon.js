import { PhononTTS } from 'phonon-tts';
import workerUrl from './phonon.worker.js?worker&url';
import wasmUrl from 'phonon-tts/wasm/phonon_tts_bg.wasm?url';
import simdWasmUrl from 'phonon-tts/wasm/phonon_tts_simd_bg.wasm?url';

import { to16k } from './pcm.js';

let loading;

function load(report) {
  loading ??= PhononTTS.load({
    model: {
      weights: { q8: '/assets/phonon/model.q8.gguf' },
      tokenizer: '/assets/phonon/tokenizer.json',
      config: '/assets/phonon/config.json',
      voices: { easton: '/assets/phonon/easton.safetensors' },
      defaultVoice: 'easton',
    },
    lang: 'en',
    quant: 'q8',
    device: 'cpu',
    threads: 1,
    workerUrl,
    wasmUrl,
    simdWasmUrl,
    onProgress: ({ total, loaded, cached }) => {
      if (total > 1_000_000 && !cached) {
        report(`Loading Phonon in your browser… ${Math.min(100, Math.round(100 * loaded / total))}%`);
      }
    },
  }).catch((error) => {
    loading = undefined;
    throw new Error(`Could not load browser Phonon. Run sh scripts/fetch_phonon.sh to install its local assets. ${error.message}`);
  });
  return loading;
}

export async function openPhononStream(text, report, signal) {
  const engine = await load(report);
  signal.throwIfAborted();
  report('Generating speech with Phonon…');
  const speech = engine.stream(text, { voice: 'easton', signal });
  return {
    chunks: to16k(speech, engine.sampleRate),
    done: speech.done,
  };
}

import { PhononTTS } from 'phonon-tts';
import workerUrl from './phonon.worker.js?worker&url';
import wasmUrl from 'phonon-tts/wasm/phonon_tts_bg.wasm?url';
import simdWasmUrl from 'phonon-tts/wasm/phonon_tts_simd_bg.wasm?url';

import { modelAssetUrl } from './model-assets.js';
import { to16k, observeFirstAudio } from './pcm.js';

let loading;

export function preparePhonon(report, mobile) {
  const pageWorkerUrl = new URL(workerUrl, location.href);
  // WebKit can reject a cached module worker script after a same-tab navigation.
  pageWorkerUrl.searchParams.set('page', String(performance.timeOrigin));
  loading ??= PhononTTS.load({
    model: {
      weights: { q8: modelAssetUrl('/assets/phonon/model.q8.gguf') },
      tokenizer: modelAssetUrl('/assets/phonon/tokenizer.json'),
      config: modelAssetUrl('/assets/phonon/config.json'),
      voices: { easton: modelAssetUrl('/assets/phonon/easton.safetensors') },
      defaultVoice: 'easton',
    },
    lang: 'en',
    quant: 'q8',
    device: 'cpu',
    threads: 1,
    // The SDK otherwise sends the downloaded weights back to the page and
    // copies them into a cache response while the worker model is live.
    cache: !mobile,
    workerUrl: pageWorkerUrl.href,
    wasmUrl,
    simdWasmUrl,
    onProgress: ({ total, loaded, cached }) => {
      if (total > 1_000_000 && !cached) {
        report(`Loading Phonon in your browser… ${Math.min(100, Math.round(100 * loaded / total))}%`);
      }
    },
  }).catch((error) => {
    loading = undefined;
    if (error.message.startsWith('phonon-tts worker failed:')) {
      throw new Error(`Could not start the Phonon worker. ${error.message}`, { cause: error });
    }
    const hint = import.meta.env.MODE === 'hosted'
      ? 'Check your connection and try again.'
      : 'For local assets, run sh scripts/fetch_phonon.sh.';
    throw new Error(`Could not load Phonon. ${hint} ${error.message}`, { cause: error });
  });
  return loading;
}

export async function releasePhonon() {
  const prepared = loading;
  loading = undefined;
  (await prepared)?.dispose();
}

export async function openPhononStream(text, report, signal, mobile, onFirstAudio) {
  const prepared = preparePhonon(report, mobile);
  const engine = await prepared;
  signal.throwIfAborted();
  report('Generating speech with Phonon…');
  const speech = engine.stream(text, { voice: 'easton', signal });
  speech.done.catch((error) => {
    if (loading === prepared && error.message.startsWith('phonon-tts worker failed:')) {
      loading = undefined;
    }
  });
  return {
    chunks: to16k(observeFirstAudio(speech, onFirstAudio), engine.sampleRate),
    done: speech.done,
  };
}

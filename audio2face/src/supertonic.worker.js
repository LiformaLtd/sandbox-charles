import * as ort from 'onnxruntime-web/wasm';
import { chunkText, loadTextToSpeech, loadVoiceStyle } from './vendor/supertonic.js';

const base = '/assets/supertonic';
let loading;

function report(message) {
  postMessage({ type: 'status', message });
}

async function load() {
  ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 1, 4);
  const style = await loadVoiceStyle([`${base}/voice_styles/M1.json`]);
  const { textToSpeech } = await loadTextToSpeech(`${base}/onnx`, { executionProviders: ['wasm'] },
    (name, index, count) => report(`Loading Supertonic 3 in your browser… ${index}/${count} · ${name}`));
  return { style, textToSpeech };
}

function emitAudio(pcm, rate) {
  postMessage({ type: 'audio', pcm, rate }, [pcm.buffer]);
}

self.onmessage = async ({ data: text }) => {
  try {
    loading ??= load();
    const { style, textToSpeech } = await loading;
    report('Generating speech with Supertonic 3…');
    const { sampleRate } = textToSpeech;
    const segments = chunkText(text, 160);
    for (const [index, segment] of segments.entries()) {
      const { wav, duration } = await textToSpeech.batch([segment], ['en'], style, 8);
      const pcm = Float32Array.from(wav.slice(0, Math.floor(duration[0] * sampleRate)));
      if (index > 0) emitAudio(new Float32Array(Math.round(0.3 * sampleRate)), sampleRate);
      if (pcm.length) emitAudio(pcm, sampleRate);
    }
    postMessage({ type: 'done' });
  } catch (error) {
    postMessage({ type: 'error', message: `Supertonic 3 failed. Check its local assets with sh scripts/fetch_onnx_tts.sh. ${error.message}` });
  }
};

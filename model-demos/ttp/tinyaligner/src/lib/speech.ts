import { PhononTTS, concatPcm, encodeWav } from 'phonon-tts';
import workerUrl from './phonon.worker.ts?worker&url';
import wasmUrl from 'phonon-tts/wasm/phonon_tts_bg.wasm?url';
import simdWasmUrl from 'phonon-tts/wasm/phonon_tts_simd_bg.wasm?url';

const voiceUrl = 'https://huggingface.co/gradium/phonon-voices/resolve/0ee17205f254b5abc43b266c8a0f215565c352f7/voices/LMDIYE0Oa377Qh6Z.safetensors';
let engine: PhononTTS | undefined;
let voiceReady = false;

export async function synthesize(text: string, signal: AbortSignal, progress: (message: string) => void) {
  if (!engine) {
    progress('Loading the speech model…');
    const loaded = await PhononTTS.load({
      lang: 'en', quant: 'q8', device: 'cpu', threads: 1, workerUrl, wasmUrl, simdWasmUrl,
      onProgress: ({ loaded, total, cached }) => {
        if (!signal.aborted && total && !cached) {
          progress(`Downloading speech assets · ${Math.min(100, Math.round(loaded / total * 100))}% of current file`);
        }
      }
    });
    if (signal.aborted) loaded.dispose();
    signal.throwIfAborted();
    engine = loaded;
  }
  if (!voiceReady) {
    progress('Loading the English voice…');
    await engine.addVoice('Easton', voiceUrl);
    signal.throwIfAborted();
    voiceReady = true;
  }
  progress('Generating speech…');
  const chunks: Float32Array[] = [];
  for await (const chunk of engine.stream(text, { voice: 'Easton', signal })) chunks.push(chunk);
  signal.throwIfAborted();
  const samples = concatPcm(chunks);
  const sampleRate = engine.sampleRate;
  return { samples, sampleRate, wav: encodeWav(samples, sampleRate) };
}

export function disposeSpeech() {
  engine?.dispose();
  engine = undefined;
  voiceReady = false;
}

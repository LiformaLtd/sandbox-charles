import PocketWorker from './vendor/pocket.worker.js?worker';
import SupertonicWorker from './supertonic.worker.js?worker';
import { to16k } from './pcm.js';

const workers = new Map();
const engines = { pocket: [PocketWorker, 24000], supertonic: [SupertonicWorker, 44100] };

export function releaseOnnxTts(engine) {
  workers.get(engine)?.terminate();
  workers.delete(engine);
}

export function openOnnxStream(engine, text, report, signal, onFirstAudio) {
  const pcm = runWorker(engine, text, report, signal, onFirstAudio);
  return {
    chunks: to16k(pcm, engines[engine][1]),
  };
}

export async function prepareOnnxTts(engine, report) {
  const pcm = runWorker(engine, null, report);
  for await (const _ of pcm) {
    // The worker acknowledges preparation by closing without emitting audio.
  }
}

function runWorker(engine, text, report, signal, onFirstAudio) {
  signal?.throwIfAborted();
  const [SpeechWorker, rate] = engines[engine];
  const worker = workers.get(engine) ?? new SpeechWorker();
  workers.set(engine, worker);

  return new ReadableStream({
    start(controller) {
      function detach() {
        worker.onmessage = null;
        worker.onerror = null;
        signal?.removeEventListener('abort', abort);
      }
      function fail(error) {
        detach();
        worker.terminate();
        workers.delete(engine);
        controller.error(error);
      }
      function abort() {
        fail(signal.reason);
      }
      worker.onmessage = ({ data }) => {
        switch (data.type) {
          case 'status':
            report(data.message);
            break;
          case 'audio':
            if (data.rate === rate) {
              if (data.pcm.length) { onFirstAudio?.(); onFirstAudio = undefined; }
              controller.enqueue(data.pcm);
            }
            else fail(new Error(`Unexpected TTS sample rate: ${data.rate} Hz.`));
            break;
          case 'done':
            detach();
            controller.close();
            break;
          case 'error':
            fail(new Error(data.message));
            break;
        }
      };
      worker.onerror = (event) => fail(new Error(event.message || 'The browser speech worker failed.'));
      signal?.addEventListener('abort', abort, { once: true });
      worker.postMessage(text);
    },
  });
}

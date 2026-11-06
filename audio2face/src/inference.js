import { modelAssetUrl } from './model-assets.js';
import { geometryBlocks } from './geometry.mjs';
import { joinHeadWindow, predictHead } from './head-motion.js';

async function downloadFaceModel(url, report) {
  report('Downloading the STA model…');
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not download the face model (HTTP ${response.status}).`);
  const length = Number(response.headers.get('content-length'));
  if (!response.body || !Number.isSafeInteger(length) || length <= 0
      || response.headers.get('content-encoding')) return response.arrayBuffer();

  const bytes = new Uint8Array(length);
  const reader = response.body.getReader();
  let offset = 0;
  let percent = -1;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes.set(value, offset);
      offset += value.byteLength;
      const next = Math.floor(100 * offset / length);
      if (next !== percent) {
        percent = next;
        report(`Downloading the STA model… ${percent}%`);
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (offset !== length) throw new Error('The face model download was incomplete.');
  return bytes;
}

let modelSession;
export async function loadModel({ cpuOnly = false, mobile = false, report } = {}) {
  if (!modelSession) {
    modelSession = (async () => {
      if (!cpuOnly && !await navigator.gpu?.requestAdapter()) {
        throw new Error('WebGPU is unavailable. Open this app in a browser with WebGPU enabled.');
      }
      const ort = cpuOnly
        ? await import('onnxruntime-web/wasm')
        : await import('onnxruntime-web/webgpu');
      ort.env.wasm.numThreads = mobile || cpuOnly ? 1 : Math.min(4, navigator.hardwareConcurrency || 1);
      const url = modelAssetUrl('/assets/model.onnx');
      const model = report ? await downloadFaceModel(url, report) : url;
      report?.(`Starting the STA model with ${cpuOnly ? 'CPU' : 'WebGPU'}…`);
      const executionProviders = cpuOnly ? ['wasm'] : ['webgpu', 'wasm'];
      const options = { executionProviders };
      if (mobile) {
        // Keep quantized constants from being expanded during session creation.
        options.extra = { optimization: { disable_specified_optimizers: 'ConstantFolding' } };
        if (!cpuOnly) {
          // Release temporary WebGPU buffers after queued work has used them.
          executionProviders[0] = {
            name: 'webgpu',
            storageBufferCacheMode: 'lazyRelease',
            defaultBufferCacheMode: 'lazyRelease',
          };
        }
      }
      try {
        const session = await ort.InferenceSession.create(model, options);
        return { ort, session };
      } finally {
        // ORT copies the download into its own memory during create(). Drop
        // our backing store before reporting ready, without waiting for GC.
        if (typeof model !== 'string') {
          const buffer = ArrayBuffer.isView(model) ? model.buffer : model;
          buffer.transfer?.(0);
        }
      }
    })().catch((error) => {
      modelSession = undefined;
      throw error;
    });
  }
  return modelSession;
}

function diffusionNoise() {
  // Seeded Box–Muller. Reuse the same noise for every speech clip.
  const values = new Float32Array(1 * 3 * 60 * 88831);
  let state = 47;
  function uniform() {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  }
  for (let i = 0; i < values.length; i += 2) {
    const amplitude = Math.sqrt(-2 * Math.log(uniform()));
    const phase = 2 * Math.PI * uniform();
    values[i] = amplitude * Math.cos(phase);
    values[i + 1] = amplitude * Math.sin(phase);
  }
  return values;
}

let noise;
export async function* animationBlocks({ ort, session }, audio, signal) {
  noise ??= diffusionNoise();
  let previous;
  const inferHead = import.meta.env.MODE === 'hosted'
    ? undefined
    : (pcm, start) => predictHead(pcm, start, signal);
  for await (const { sampleIndices, geometry, pcm, head } of geometryBlocks(ort, session, audio, noise, 1, inferHead)) {
    const block = geometry.slice();
    const frames = sampleIndices.map((_, i) => block.subarray(i * 88831, (i + 1) * 88831));
    const rotations = head ? joinHeadWindow(head, previous, sampleIndices) : sampleIndices.map(() => undefined);
    previous = head;
    yield { frames, pcm, rotations };
  }
}

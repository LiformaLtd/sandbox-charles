import { modelAssetUrl } from './model-assets.js';
import { audioWindows, geometryBlocks } from './geometry.mjs';
import { joinHeadWindow, predictHead } from './head-motion.js';

async function downloadFaceModel(url, report) {
  report?.('Downloading the STA model…');
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
        report?.(`Downloading the STA model… ${percent}%`);
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (offset !== length) throw new Error('The face model download was incomplete.');
  return bytes;
}

const modelPaths = {
  geometry: '/assets/model.onnx',
  compact: '/assets/james-arkit-student.onnx',
  'compact-v2': '/assets/james-arkit-student-v2.onnx',
  'compact-v3': '/assets/james-arkit-student-v3.onnx',
  'compact-v4': '/assets/james-arkit-student-v4.onnx',
};

let modelSession;
export async function loadModel({ model = 'geometry', cpuOnly = false, mobile = false, report } = {}) {
  if (!modelSession) {
    modelSession = (async () => {
      const adapter = cpuOnly ? null : await navigator.gpu?.requestAdapter();
      if (!cpuOnly && !adapter) {
        throw new Error('WebGPU is unavailable. Open this app in a browser with WebGPU enabled.');
      }
      const url = modelAssetUrl(modelPaths[model]);
      if (!cpuOnly && ['compact-v2', 'compact-v3', 'compact-v4'].includes(model)) {
        const { createCompactWebGpu } = await import('./compact-webgpu.js');
        const data = await downloadFaceModel(url, report);
        report?.('Starting the STA model with WebGPU…');
        try {
          const run = await createCompactWebGpu(adapter, data, () => { modelSession = undefined; });
          return { compact: true, runCompact: async (samples) => ({ data: await run(samples) }) };
        } finally {
          (data.buffer ?? data).transfer?.(0);
        }
      }
      const ort = cpuOnly
        ? await import('onnxruntime-web/wasm')
        : await import('onnxruntime-web/webgpu');
      ort.env.wasm.numThreads = mobile || cpuOnly ? 1 : Math.min(4, navigator.hardwareConcurrency || 1);
      const modelData = report ? await downloadFaceModel(url, report) : url;
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
        const session = await ort.InferenceSession.create(modelData, options);
        if (model === 'geometry') return { ort, session, compact: false };
        return {
          compact: true,
          async runCompact(samples) {
            const input = new ort.Tensor('float32', samples, [1, 16000]);
            try {
              return (await session.run({ pcm: input })).blendshapes;
            } finally {
              input.dispose();
            }
          },
        };
      } finally {
        // ORT copies the download into its own memory during create(). Drop
        // our backing store before reporting ready, without waiting for GC.
        if (typeof modelData !== 'string') {
          const buffer = ArrayBuffer.isView(modelData) ? modelData.buffer : modelData;
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
async function* blendshapeBlocks(runCompact, audio, signal, inferHead) {
  for await (const window of audioWindows(audio instanceof Float32Array ? [audio] : audio)) {
    const { samples, sampleIndices, first, pcm, headWindow, headStart } = window;
    if (!sampleIndices.length) continue;
    signal.throwIfAborted();
    const pendingHead = inferHead?.(headWindow, headStart).then(
      (track) => ({ track }), (error) => ({ error }),
    );
    let output;
    try {
      output = await runCompact(samples);
      const head = await pendingHead;
      if (head?.error) throw head.error;
      yield {
        sampleIndices,
        pcm,
        head: head?.track,
        geometry: output.data.subarray(first * 52, (first + sampleIndices.length) * 52),
      };
    } finally {
      output?.dispose?.();
    }
  }
}

export async function* animationBlocks({ ort, session, compact, runCompact }, audio, signal) {
  let previous;
  const inferHead = import.meta.env.MODE === 'hosted'
    ? undefined
    : (pcm, start) => predictHead(pcm, start, signal);
  const source = compact
    ? blendshapeBlocks(runCompact, audio, signal, inferHead)
    : geometryBlocks(ort, session, audio, noise ??= diffusionNoise(), 1, inferHead);
  const width = compact ? 52 : 88831;
  for await (const { sampleIndices, geometry, pcm, head } of source) {
    const block = geometry.slice();
    const frames = sampleIndices.map((_, i) => block.subarray(i * width, (i + 1) * width));
    const rotations = head ? joinHeadWindow(head, previous, sampleIndices) : sampleIndices.map(() => undefined);
    previous = head;
    yield { frames, pcm, rotations };
  }
}

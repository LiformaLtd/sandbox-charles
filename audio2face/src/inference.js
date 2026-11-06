import * as ort from 'onnxruntime-web/webgpu';
import { geometryBlocks } from './geometry.mjs';
import { joinHeadWindow, predictHead } from './head-motion.js';

let modelSession;
export async function loadModel() {
  if (!modelSession) {
    modelSession = (async () => {
      if (!await navigator.gpu?.requestAdapter()) throw new Error('WebGPU is unavailable. Open this app in Chrome with graphics acceleration enabled.');
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
      return ort.InferenceSession.create('/assets/model.onnx', { executionProviders: ['webgpu', 'wasm'] });
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
export async function* animationBlocks(session, audio, signal) {
  noise ??= diffusionNoise();
  let previous;
  const inferHead = (pcm, start) => predictHead(pcm, start, signal);
  for await (const { sampleIndices, geometry, pcm, head } of geometryBlocks(ort, session, audio, noise, 1, inferHead)) {
    const block = geometry.slice();
    const frames = sampleIndices.map((_, i) => block.subarray(i * 88831, (i + 1) * 88831));
    const rotations = joinHeadWindow(head, previous, sampleIndices);
    previous = head;
    yield { frames, pcm, rotations };
  }
}

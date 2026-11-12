/// <reference lib="webworker" />
import wasmModule from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import wasmBinary from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { createCompactWebGpu } from './vendor/compact-webgpu';
import { windowFrames } from './windows';
import type { Runtime } from './types';

type Runner = { run: (samples: Float32Array) => Promise<Float32Array>; dispose: () => void | Promise<void> };
let cached: { key: string; value: Promise<Runner> } | undefined;
async function load(url: string, checksum: string, runtime: Runtime): Promise<Runner> {
  const key = `${url}:${checksum}:${runtime}`;
  if (cached?.key === key) return cached.value;
  if (cached) await (await cached.value).dispose();
  const value = (async () => {
    postMessage({ type: 'progress', message: 'Loading the compact shape model…', percent: 0 });
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load the shape model (HTTP ${response.status}).`);
    const data = await response.arrayBuffer();
    const digest = await crypto.subtle.digest('SHA-256',data);
    const actual = [...new Uint8Array(digest)].map(v => v.toString(16).padStart(2,'0')).join('');
    if (actual !== checksum) throw new Error('Shape model checksum mismatch. Prepare the sandbox assets again.');
    if (runtime === 'webgpu') {
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('WebGPU is unavailable. Select CPU in this visual sandbox to continue.');
      const run = await createCompactWebGpu(adapter,data,() => { cached = undefined; });
      return { run, dispose: run.dispose };
    }
    const ort = await import('onnxruntime-web/wasm');
    ort.env.wasm.numThreads = 1;ort.env.wasm.wasmPaths = { mjs: wasmModule, wasm: wasmBinary };
    const session = await ort.InferenceSession.create(data,{ executionProviders: ['wasm'] });
    return {
      async run(samples: Float32Array) {
        const tensor = new ort.Tensor('float32',samples,[1,16000]);
        try {
          const output = (await session.run({ pcm: tensor })).blendshapes;
          try { return Float32Array.from(output.data as Float32Array); }
          finally { output.dispose(); }
        } finally { tensor.dispose(); }
      },
      dispose: () => session.release()
    };
  })();
  cached = { key, value };
  try { return await value; } catch (error) { cached = undefined; throw error; }
}

self.onmessage = async ({ data }: MessageEvent<{ pcm: Float32Array; url: string; checksum: string; runtime: Runtime }>) => {
  try {
    const { pcm } = data;
    if (!pcm.length || pcm.length > 480000 || !pcm.every(Number.isFinite)) throw new Error('Expected finite mono audio, up to 30 seconds.');
    const runner = await load(data.url,data.checksum,data.runtime);
    const count = Math.ceil(pcm.length * 60 / 16000);const track = new Float32Array(count * 52);
    const seen = new Uint8Array(count);let elapsed = 0;
    for (let start = -16000; start < pcm.length - 4000; start += 8000) {
      const frames = windowFrames(start,pcm.length);if (!frames.length) continue;
      const window = new Float32Array(16000);const left = Math.max(0,start);const right = Math.min(start+16000,pcm.length);
      window.set(pcm.subarray(left,right),left-start);
      const before = performance.now();const output = await runner.run(window);elapsed += performance.now()-before;
      if (output.length !== 60 * 52 || !output.every(Number.isFinite)) throw new Error('Unexpected shape model output.');
      for (const f of frames) { track.set(output.subarray(f.local*52,(f.local+1)*52),f.global*52);seen[f.global]++; }
      postMessage({ type: 'progress', message: 'Preparing speech movement…', percent: Math.round((frames.at(-1)!.global+1)/count*100) });
    }
    if (!seen.every(v => v === 1)) throw new Error('The shape track has missing or overlapping audio frames.');
    postMessage({ type: 'complete', frames: track, count, inferenceMs: elapsed, duration: pcm.length/16000 }, { transfer: [track.buffer] });
  } catch (error) { postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) }); }
};

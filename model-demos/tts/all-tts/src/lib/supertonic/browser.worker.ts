import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import wasmModule from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import { loadEngine, type Engine } from './engine';
import { revision, type Settings, type Packet } from './model';
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: wasmModule };
let engine: Engine | undefined;
const emit = (packet: Packet) => {
 if (packet.pcm) self.postMessage(packet, { transfer: [packet.pcm.buffer] });
 else self.postMessage(packet);
};
async function asset(file: string): Promise<Uint8Array<ArrayBuffer>> {
 const url = `/api/assets/${file}`;
 const cache = await caches.open(`supertonic-3-${revision}`);
 let response = await cache.match(url);
 if (!response) {
  emit({ status: `Downloading ${file.split('/').pop()}…` });
  response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${file} (${response.status}).`);
  await cache.put(url, response.clone());
 }
 return new Uint8Array(await response.arrayBuffer());
}
self.onmessage = async ({ data }: MessageEvent<Settings>) => {
 const started = performance.now();
 try {
  if (!engine) engine = await loadEngine(ort, asset, status => emit({ status }), 'wasm');
  await engine.prepare(data.voice);
  emit({ setup: performance.now() - started });
  await engine.generate(data, emit, () => false);
 } catch (e) { console.error('Supertonic browser synthesis failed', e); emit({ error: e instanceof Error ? e.message : String(e) }); }
};

import wasmModule from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import wasmBinary from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';

export function configureTtsWasm(ort) {
  ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 1, 4);
  // In the bundled TTS worker, ORT needs its own module for its child threads.
  ort.env.wasm.wasmPaths = {
    mjs: new URL(wasmModule, self.location.href).href,
    wasm: new URL(wasmBinary, self.location.href).href,
  };
}

import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
const headers={'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'};
export default defineConfig({
 worker:{format:'es'},optimizeDeps:{include:['onnxruntime-web/wasm','phonon-tts/worker.js','protobufjs/minimal.js']},
 server:{headers},preview:{headers},plugins:[sveltekit({adapter:adapter()})]
});

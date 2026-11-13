import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { createHeadPoseServer } from './scripts/head-pose-server.js';

const root = fileURLToPath(new URL('.', import.meta.url));
const assets = {
  '/assets/model.onnx': ['model.onnx', 'application/octet-stream'],
  '/assets/james-arkit-student.onnx': ['arkit-student/james-arkit-student.onnx', 'application/octet-stream'],
  '/assets/james-arkit-student-v2.onnx': ['arkit-student/james-arkit-student-v2.onnx', 'application/octet-stream'],
  '/assets/james-arkit-student-v3.onnx': ['arkit-student/james-arkit-student-v3.onnx', 'application/octet-stream'],
  '/assets/james-arkit-student-v4.onnx': ['arkit-student/james-arkit-student-v4.onnx', 'application/octet-stream'],
  '/assets/james-blendshapes.bin': ['arkit-student/james-blendshapes.bin', 'application/octet-stream'],
  '/assets/rig.bin': ['rig.bin', 'application/octet-stream'],
  '/assets/rig.json': ['rig.json', 'application/json'],
  '/assets/phonon/model.q8.gguf': ['phonon/model.q8.gguf', 'application/octet-stream'],
  '/assets/phonon/tokenizer.json': ['phonon/tokenizer.json', 'application/json'],
  '/assets/phonon/config.json': ['phonon/config.json', 'application/json'],
  '/assets/phonon/easton.safetensors': ['phonon/easton.safetensors', 'application/octet-stream'],
  '/assets/supertonic/onnx/duration_predictor.onnx': ['supertonic/onnx/duration_predictor.onnx', 'application/octet-stream'],
  '/assets/supertonic/onnx/text_encoder.onnx': ['supertonic/onnx/text_encoder.onnx', 'application/octet-stream'],
  '/assets/supertonic/onnx/vector_estimator.onnx': ['supertonic/onnx/vector_estimator.onnx', 'application/octet-stream'],
  '/assets/supertonic/onnx/vocoder.onnx': ['supertonic/onnx/vocoder.onnx', 'application/octet-stream'],
  '/assets/supertonic/onnx/tts.json': ['supertonic/onnx/tts.json', 'application/json'],
  '/assets/supertonic/onnx/unicode_indexer.json': ['supertonic/onnx/unicode_indexer.json', 'application/json'],
  '/assets/supertonic/voice_styles/M1.json': ['supertonic/voice_styles/M1.json', 'application/json'],
  '/assets/pocket/bundle.json': ['pocket/bundle.json', 'application/json'],
  '/assets/pocket/flow_lm_flow_int8.onnx': ['pocket/flow_lm_flow_int8.onnx', 'application/octet-stream'],
  '/assets/pocket/flow_lm_main_int8.onnx': ['pocket/flow_lm_main_int8.onnx', 'application/octet-stream'],
  '/assets/pocket/mimi_decoder_int8.onnx': ['pocket/mimi_decoder_int8.onnx', 'application/octet-stream'],
  '/assets/pocket/text_conditioner_int8.onnx': ['pocket/text_conditioner_int8.onnx', 'application/octet-stream'],
  '/assets/pocket/tokenizer.model': ['pocket/tokenizer.model', 'application/octet-stream'],
  '/assets/pocket/voices.bin': ['pocket/voices.bin', 'application/octet-stream'],
  '/assets/pocket/sentencepiece.js': ['pocket/sentencepiece.js', 'text/javascript'],
};

function fail(response, status, message) {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: message }));
}

function localStudio() {
  function configure(server) {
    const headPose = createHeadPoseServer(root);
    server.httpServer?.once('close', headPose.close);
    server.middlewares.use(async (request, response, next) => {
      const pathname = request.url?.split('?')[0];
      try {
        if (pathname === '/api/idle-head-motion') {
          if (request.method !== 'GET') return fail(response, 405, 'Use GET to load idle head motion.');
          const { fps, angles } = await headPose.idle();
          response.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
          return response.end(JSON.stringify({ fps, angles }));
        }
        if (pathname === '/api/head-motion') {
          if (request.method !== 'POST') return fail(response, 405, 'Use POST to generate head motion.');
          if (request.headers['content-type'] !== 'application/octet-stream') {
            return fail(response, 415, 'Send 16 kHz float PCM.');
          }
          const startSample = Number(request.headers['x-start-sample']);
          if (!Number.isSafeInteger(startSample)) return fail(response, 400, 'Invalid audio position.');
          let size = 0;
          const chunks = [];
          for await (const part of request) {
            size += part.length;
            if (size > 128_000) return fail(response, 413, 'The audio window is too large.');
            chunks.push(part);
          }
          if (!size || size % 4) return fail(response, 400, 'Invalid PCM window.');
          const { fps, angles } = await headPose.predict(Buffer.concat(chunks), startSample);
          response.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
          return response.end(JSON.stringify({ fps, angles }));
        }
        const asset = Object.hasOwn(assets, pathname) ? assets[pathname] : undefined;
        if (!asset) return next();
        if (request.method !== 'GET' && request.method !== 'HEAD') return fail(response, 405, 'Use GET or HEAD.');
        const [name, contentType] = asset;
        const file = join(root, '.local', name);
        const info = await stat(file);
        response.writeHead(200, {
          'Content-Type': contentType,
          'Content-Length': info.size,
          'Cache-Control': 'no-cache',
        });
        if (request.method === 'HEAD') return response.end();
        await pipeline(createReadStream(file), response);
      } catch (error) {
        console.error(pathname, error.message);
        if (response.headersSent) return response.destroy();
        if (error.code === 'ENOENT') {
          return fail(response, 404, 'Local model or rig asset is missing. See the setup instructions.');
        }
        if (pathname === '/api/head-motion' || pathname === '/api/idle-head-motion') {
          return fail(response, 500, error.message);
        }
        fail(response, 500, 'Could not serve the local asset.');
      }
    });
  }
  return { name: 'james-local-studio', configureServer: configure, configurePreviewServer: configure };
}

export default defineConfig(() => {
  const headers = {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  };
  return {
    plugins: [localStudio()],
    server: { host: '127.0.0.1', port: 5173, strictPort: false, headers },
    preview: { host: '127.0.0.1', port: 5173, strictPort: false, headers },
    worker: { format: 'es' },
    optimizeDeps: { exclude: ['onnxruntime-web/webgpu', 'onnxruntime-web/wasm'] },
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.endsWith('/three/build/three.core.js')) return 'three-core';
          },
        },
      },
    },
  };
});

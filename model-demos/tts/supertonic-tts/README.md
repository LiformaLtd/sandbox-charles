# Supertonic 3 Lab

SvelteKit 3 lab comparing native local-server and browser ONNX inference with the same Supertonic 3 standard FP32 model files, ten preset voices, 31 languages, speed and denoising steps. Text examples match the Pocket and Phonon labs.

## Run

Requires Node 22.17 or newer. From this folder:

```sh
npm ci
npm run dev
```

Open http://127.0.0.1:5182/. The SvelteKit server itself runs native `onnxruntime-node` on CPU; no separate Python service is needed. Browser mode uses `onnxruntime-web` WASM in a worker. Both use one CPU thread for comparison. The first run downloads about 400 MB from the pinned public model revision, then caches them under `~/Source/liforma/ai-workflows/apple-silicon/.supertonic-3-cache/3cadd1ee6394adea1bd021217a0e650ede09a323`. Set `SUPERTONIC_MODEL_DIR` to use another directory containing `onnx/` and `voice_styles/`. Do not commit weights or private audio.

Browser mode downloads from the local asset route and keeps a separate copy in browser Cache Storage. After the initial download, synthesis has no dependency on a cloud inference API. The local app still needs to be running to serve the page.

Supertonic generates complete segments, rather than producing autoregressive audio frames. This lab splits sentences (and bounds long sentences) identically in both modes and plays each segment as soon as it completes, with 0.3 seconds between segments. TTFA starts on Generate and includes model setup, download time, queueing and first-segment synthesis. Total generation measures all segment synthesis after setup; playback can continue after generation finishes. Speed includes the inserted gaps. Stop cancels between segments on the server and terminates the browser worker; stopping browser inference requires reloading its model on the next run.

`npm run check` and `npm run build` validate the app. Serve the Node production build with `HOST=127.0.0.1 PORT=5182 node build`.

## Upstream

- Model: https://huggingface.co/Supertone/supertonic-3 at `3cadd1ee6394adea1bd021217a0e650ede09a323`. Model assets use the upstream OpenRAIL licence; see its LICENSE for conditions.
- Inference helper: https://github.com/supertone-oss-archive/supertonic at `1e9799e964ea4c0dad7cde993b65c3c813a7b373`, MIT (notice included in `src/lib/vendor/LICENSE`). Adapted to inject the ONNX runtime so both modes share the same inference implementation.

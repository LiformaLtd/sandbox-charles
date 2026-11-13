# All TTS Lab

A shared SvelteKit 3 test bench for Phonon, Supertonic and Pocket TTS. Pick a family, then a model, runtime and voice. Text examples match the individual labs. Audio plays while generation continues, with TTFA, setup time, total generation time, audio duration, generation speed and WAV download.

| Family | Models | Runtime |
| --- | --- | --- |
| Phonon | 31M English | Browser WASM Q8, local native CPU/Q8, local MLX Apple GPU |
| Supertonic | Supertonic 3, standard 99M | Native ONNX on local app server or browser ONNX WASM |
| Pocket TTS | 1.9M, 6.9M, 18.2M, original 100M | Local Python; browser ONNX for 18.2M REFL |

Phonon has all 37 English catalogue voices. Supertonic has ten presets and 31 language choices. Pocket has Alba/Anna/Marius locally, eight presets in browser ONNX, and WAV/M4A reference cloning in both modes (up to 10 MB, first 30 seconds). Browser controls only offer models/runtimes that are implemented. Non-English Phonon checkpoints are outside this demo's current scope. Phonon 40M is gated on Hugging Face and the browser library rejects gated repositories, so it is not enabled in the selector.

## Run

Requires Node 22.17 or newer:

```sh
npm ci
npm run dev
```

Open the URL printed by Vite: port 5173, or the first free port after it. Supertonic native inference is part of this Node app, without a separate service. Phonon uses its official WASM worker; the ONNX modes use ONNX Runtime Web WASM. Browser runs use a single CPU thread.

Pocket local mode deliberately shares the existing Pocket Python service at http://127.0.0.1:8178; set `POCKET_TTS_URL` for another endpoint. To run a separate service from this app, install uv and Python 3.12:

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python -m uvicorn inference:app --host 127.0.0.1 --port 8178
```

Do not run two Pocket services on the same port. The included inference service is self-contained; it does not import sibling app code. Local cloning needs access to `kyutai/pocket-tts` on Hugging Face and FFmpeg for M4A. Authorise access and sign in using `.venv/bin/hf auth login` if needed.

## Timing and caching

TTFA starts on Generate and ends when the first playable audio is scheduled. It includes model downloads/loading and voice preparation. Total generation measures synthesis after setup. The families use different voices and runtime implementations; speed is not an isolated model-quality comparison. Supertonic generates complete text segments before playback, with 0.3-second gaps between segments; Phonon and Pocket deliver smaller streaming chunks. Stopping browser ONNX generation terminates its worker, so the next run reloads sessions from cached assets. Model changes replace the active Phonon instance to bound memory use.

Phonon and Pocket ONNX cache downloads in browser storage. Pocket Python uses `~/Source/liforma/ai-workflows/apple-silicon/.pocket-tts-cache`. Supertonic uses `~/Source/liforma/ai-workflows/apple-silicon/.supertonic-3-cache/<revision>`, sharing the same pinned assets as its individual demo; override with `SUPERTONIC_MODEL_DIR`. Supertonic's initial model download is about 400 MB. No private voice files or model weights belong in Git.

## Model sources and licences

Model registry: `src/lib/catalogue.ts`. Pinned revisions prevent silent changes to the browser checkpoints. Phonon 31M and its public voice catalogue are Apache 2.0; The currently unavailable Phonon 40M is CC BY-NC 4.0. Supertonic 3 uses its upstream OpenRAIL licence. Pocket models and the ONNX conversion use CC BY 4.0. Keep upstream licence conditions and voice-specific permissions when redistributing.

- https://huggingface.co/gradium/phonon-31m-en
- https://huggingface.co/gradium/phonon-40m-en
- https://huggingface.co/gradium/phonon-voices
- https://huggingface.co/Supertone/supertonic-3
- https://huggingface.co/kyutai/pocket-tts
- https://huggingface.co/maxime/pocket-tts-18m-onnx

Supertonic inference is adapted from `supertone-oss-archive/supertonic` at `1e9799e964ea4c0dad7cde993b65c3c813a7b373`, MIT; notice included under `src/lib/supertonic/vendor/LICENSE`.

Validate with `npm run check` and `npm run build`. The local production build uses `npm run preview`. Its origin is derived from the incoming request.

## Phonon local runtimes

For Phonon 31M, **Run on** offers Browser, Local native (CPU/Q8) and Local MLX (Apple GPU, standard weights). Local modes use the isolated service in `ai-workflows/apple-silicon/phonon`. Run `./phonon/run.sh` there to serve port 8184. Configure `PHONON_TTS_URL` to override the service address. This dependency is deliberate; no sibling app server is needed.

The `/api/phonon` proxy carries the service's incremental float32 audio stream to the existing playback/timing controls. TTFA includes the request and first scheduled playback; setup/generation times come from the selected service backend. All 37 English catalogue voices are available in all three Phonon runtimes. Native, MLX and browser use different devices/formats/threading, so compare them as complete runtime paths.

## Browser tokenizer build and regression check

`npm ci` builds a browser-only engine from the exact `@sctg/sentencepiece-js` 1.3.3 source. The build verifies its SHA-256, extracts the upstream Emscripten engine, selects its browser platform and rejects any remaining imports of Node modules. Generated JavaScript and embedded WebAssembly stay out of Git. The small adapter accepts the existing tokenizer bytes directly; token IDs are unchanged. Upstream source: https://github.com/sctg-development/sentencepiece-js (Apache-2.0; notice in `src/lib/pocket/onnx/generated/LICENSE`).

Run `npm run test:tokenizer` to render the lab in Chrome and compare 32 tokenization cases against the original wrapper. It downloads only the pinned public Pocket tokenizer and checks its SHA-256; no speech weights, voice data or recordings are downloaded. Chrome is expected at its standard macOS path; set `CHROME_EXECUTABLE` elsewhere. The test starts its own server at port 5173 or the first free port and uses its printed URL. Full speech inference still requires the model/runtime selected in the UI.

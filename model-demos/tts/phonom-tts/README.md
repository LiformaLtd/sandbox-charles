# Phonon TTS Lab

Compare the clarity and speed of Gradium Phonon 31M English with Pocket TTS. The opening prompt matches the Pocket TTS Lab. The other prompts exercise numbers, units and consonants. Streaming playback starts when the first audio chunk arrives.

Requires Node 22.17 or newer and a browser with WebAssembly SIMD.

```sh
npm ci
npm run dev
```

Open the URL printed by Vite: port 5173, or the first free port after it. Browser mode’s first run downloads the public Q8 model (64.5 MB of weights), tokenizer, and voice assets. In Browser mode, the browser caches those files and runs synthesis locally using the official Phonon WASM worker. This lab fixes the CPU runtime at one thread for repeatable comparisons. The picker includes all 37 English presets from the [Gradium voice catalogue](https://huggingface.co/gradium/phonon-voices). Each preset downloads when first used.

The **Time to first audio** measurement starts on Generate and includes loading or downloading assets. **Total generation** is the worker's synthesis time after setup. Generation speed divides clip duration by that time. The finished recording can be replayed or downloaded as a WAV.

Run `npm run check` and `npm run build` to validate. Run the production build with `npm run preview`.

[Gradium Phonon 31M English](https://huggingface.co/gradium/phonon-31m-en) · [Official browser demo](https://huggingface.co/spaces/gradium/phonon)

Place downloaded Phonon `.safetensors` voice files in `src/lib/local-voices/` to list them under **Local voices**. The filename becomes the picker label. The directory is ignored by Git. A build made with those files present includes them as browser assets.

## Local native and MLX

Use **Run on** to choose Browser WASM, Local native (official Rust runtime, CPU/Q8), or Local MLX (Apple GPU, standard weights). Both local modes use the existing Phonon service in `ai-workflows/apple-silicon/phonon`, at http://127.0.0.1:8184. Start it there with `./phonon/run.sh`. Set `PHONON_TTS_URL` in this app’s environment to use another service address.

The app proxies streaming audio through `/api/phonon`; there are no direct cross-origin browser requests to the service. All 37 English presets work in each mode. Downloaded custom voices in `src/lib/local-voices` remain available in Browser mode; local service modes only list its built-in catalogue voices. Runtime and voice are recorded with each result.

TTFA is measured by the browser through first scheduled audio, including the local round trip. Generation/setup times for native and MLX come from the service. Native CPU, browser WASM, and MLX GPU use different implementations, weight formats and threading, so timings compare the complete runtime paths. Models and voices are already stored on the service; local modes download nothing during generation.

The app derives its origin from the incoming request; no fixed frontend origin is built in.

## Browser ONNX versions

The **Run on** menu also offers five ONNX conversions of **Phonon 31M English**:

| Option | Graphs + external weights |
| --- | ---: |
| Full precision · FP32 | 165.8 MB |
| Half precision · FP16 | 83.0 MB |
| INT8 | 64.0 MB |
| Q4 | 51.3 MB |
| Q4 with FP16 | 34.6 MB |

Choose **Browser CPU (WASM)** or **Browser GPU (WebGPU)** separately. These use the
same model files; WebGPU requires a browser/device with WebGPU support, and FP16
and Q4+FP16 additionally require `shader-f16`. An unsupported choice produces an
error rather than silently changing the selected version or device. ONNX Runtime
may place unsupported or shape operations on CPU even with WebGPU selected.
Smaller downloads do not guarantee faster synthesis: Q4 can be slower than FP32
on CPU. All 37 English preset voices are available. Custom `.safetensors` voices
remain exclusive to the original Browser WASM mode.

The app pins [the community ONNX conversion](https://huggingface.co/onnx-community/phonon-31m-en-ONNX)
to revision `f0f522a750bf515227cb478abb97b2ffca7e6d8c`. It loads the generator and
Mimi decoder graphs with ONNX Runtime Web, plus the matching tokenizer and `.bin`
voice files. Graphs, external weights and voices are cached in browser Cache
Storage when available; tokenizer assets use the Transformers.js cache. These
sizes exclude voices and runtime software. There is no hosted inference service
in the ONNX path and no model weights are committed to this repository.

The selected ONNX worker and inference sessions remain loaded between **Generate**
clicks. Changing version/device, leaving ONNX mode or cancelling generation
releases that worker. Only one ONNX version is held in memory at a time. Cold
setup includes downloads and graph preparation; warm setup normally approaches
zero. Each request recomputes text and voice conditioning. **Replay** uses the
already-generated WAV and performs no inference.

ONNX emits audio after each eight-frame generator block (up to 640 ms of speech),
rather than waiting for a complete sentence. The causal decoder exposes no state
cache, so the app decodes the accumulated prefix and emits only new samples;
this trades some throughput for earlier playback. GPU feedback tensors stay on
the GPU between generator calls. The official Q8 runtime streams smaller frames,
so its first-audio latency is not directly equivalent. Long text is divided at
sentence boundaries and at 40-word limits. Stop cancels downloads/inference by
terminating the ONNX worker; the next request loads fresh sessions from cache.

## Language-specific and multilingual 40M models

The **Model** menu now includes 31M English plus 40M English, French, German,
Spanish, Portuguese and multilingual. Each 40M Q8 weights file is 74.6 MB.
The multilingual model uses one weights file for all five languages; choose
**Speech language** so numbers and symbols use the correct text normalization.
Voice choices follow that language: 37 English, 15 French, 13 German, 25 Spanish
and 12 Portuguese presets (102 total). Matching sample paragraphs are included.

40M runs through **Browser Q8**, with an explicit CPU/WebGPU choice. ONNX and the
existing native/MLX service remain 31M English only. Changing model, language or
Q8 device replaces the browser engine; warm requests reuse it. WAV filenames and
result labels identify the selected model and runtime.

The official 40M repositories require accepting their Hugging Face access
conditions. Once access is enabled and the saved local token is authorised:

```sh
npm run fetch:models                  # all six 40M checkpoints
npm run fetch:models -- 40m-fr        # just French
```

The downloader uses `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or the locally saved
Hugging Face token (`HF_TOKEN_PATH` / `HF_HOME` are honoured). Never put a token
in browser code. It downloads the pinned config, tokenizer and Q8 weights into
`.models/phonon/<model>/<revision>/`, outside Git and the production bundle.
Interrupted downloads use temporary files and are retried on the next run.
A model is enabled only after all three files exist and the weights size matches.
Use **Refresh model availability** after downloading, or reload the page.

The local app serves those files to the browser through a restricted asset route;
synthesis still happens in the browser. Set `PHONON_MODEL_DIR` to relocate the
model store, using the same absolute directory for downloading and serving.
Keep that store with the local Node app when running a production build. Do not
publish or redistribute downloaded models without following their licence and
access conditions. The 31M public download and its five ONNX options continue to
work without these local files.

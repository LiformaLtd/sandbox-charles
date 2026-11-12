# Phonon TTS Lab

Compare the clarity and speed of Gradium Phonon 31M English with Pocket TTS. The opening prompt matches the Pocket TTS Lab. The other prompts exercise numbers, units and consonants. Streaming playback starts when the first audio chunk arrives.

Requires Node 22.17 or newer and a browser with WebAssembly SIMD.

```sh
npm ci
npm run dev
```

Open http://127.0.0.1:5180/. Browser mode’s first run downloads the public Q8 model (64.5 MB of weights), tokenizer, and voice assets. In Browser mode, the browser caches those files and runs synthesis locally using the official Phonon WASM worker. This lab fixes the CPU runtime at one thread for repeatable comparisons. The picker includes all 37 English presets from the [Gradium voice catalogue](https://huggingface.co/gradium/phonon-voices). Each preset downloads when first used.

The **Time to first audio** measurement starts on Generate and includes loading or downloading assets. **Total generation** is the worker's synthesis time after setup. Generation speed divides clip duration by that time. The finished recording can be replayed or downloaded as a WAV.

Run `npm run check` and `npm run build` to validate. Run the production build with `HOST=127.0.0.1 PORT=5180 node build`.

[Gradium Phonon 31M English](https://huggingface.co/gradium/phonon-31m-en) · [Official browser demo](https://huggingface.co/spaces/gradium/phonon)

Place downloaded Phonon `.safetensors` voice files in `src/lib/local-voices/` to list them under **Local voices**. The filename becomes the picker label. The directory is ignored by Git. A build made with those files present includes them as browser assets.

## Local native and MLX

Use **Run on** to choose Browser WASM, Local native (official Rust runtime, CPU/Q8), or Local MLX (Apple GPU, standard weights). Both local modes use the existing Phonon service in `ai-workflows/apple-silicon/phonon`, at http://127.0.0.1:8184. Start it there with `./phonon/run.sh`. Set `PHONON_TTS_URL` in this app’s environment to use another service address.

The app proxies streaming audio through `/api/phonon`; there are no direct cross-origin browser requests to the service. All 37 English presets work in each mode. Downloaded custom voices in `src/lib/local-voices` remain available in Browser mode; local service modes only list its built-in catalogue voices. Runtime and voice are recorded with each result.

TTFA is measured by the browser through first scheduled audio, including the local round trip. Generation/setup times for native and MLX come from the service. Native CPU, browser WASM, and MLX GPU use different implementations, weight formats and threading, so timings compare the complete runtime paths. Models and voices are already stored on the service; local modes download nothing during generation.

The production build is configured for http://127.0.0.1:5180. Change `paths.origin` in `vite.config.ts` before building for another origin.

# TinyAligner Fast

A standalone SvelteKit demo that generates an English sample with Phonon on page load and runs TinyAligner on the resulting PCM. Edit the sentence and select **Generate & align** to repeat the flow. Play the clip to follow the waveform and highlighted phonemes, or select a word or phoneme to seek to its estimated onset.

## Run locally

Requires Node 22.17 or newer and a browser with WebAssembly SIMD.

```sh
npm ci
npm run assets
npm run dev
```

Open the URL printed by Vite: port 5173, or the first free port after it. There is no required environment configuration. The asset command fetches the pinned TinyAligner model and prepares its English dictionary. The app downloads the Phonon Q8 weights (about 65 MB), tokenizer, and voice from Hugging Face on the first run. Synthesis and alignment run on the device. The text and audio are processed locally.

```sh
npm run check
npm run build
npm run preview
```

## Timing and input

Use short English sentences with dictionary pronunciations and spell out numbers. The display uses stress-free ARPABET labels and estimated CTC boundaries on a 20 ms grid. The alignment timer includes resampling, feature extraction, inference, and decoding. The first alignment also includes ONNX session setup and loading the model.

The alignment feature extraction and CTC decoder are adapted from [TinyAligner](https://github.com/krmanik/tiny-aligner/tree/1f07ad6d186c70baac4fae26c605cc84660d65f7), whose README identifies its code license as MIT. The English pronunciation data comes from CMUdict. Speech synthesis uses [Gradium Phonon](https://gradium.ai/phonon) with the Easton preset. This app has no dependency on the sibling TTS demo.

## Benchmark

The alignment pipeline is 16 kHz PCM → log-Mel → ONNX Runtime → CTC. The pinned graph contains three convolutions, one GRU node and an output projection. Phonon runs in a separate worker in the demo. The benchmark isolates alignment and times feature extraction (including input tensor creation), inference, CTC and total separately. It reports session initialization separately and excludes synthesis, resampling, asset downloads and rendering from the stage timings.

Create a mono 16 kHz little-endian float32 fixture with the phrase “The quick brown fox jumps over the lazy dog”. On macOS with FFmpeg:

```sh
fixture_dir=$(mktemp -d)
say -v Samantha -r 155 -o "$fixture_dir/fixture.aiff" 'The quick brown fox jumps over the lazy dog.'
ffmpeg -v error -i "$fixture_dir/fixture.aiff" -af apad=pad_dur=0.25 -ac 1 -ar 16000 -f f32le static/models/benchmark.f32
npm run benchmark
```

The harness repeats the same PCM and transcript 1, 4 and 12 times. Use the same fixture for every comparison. Each case warms up before reporting median wall times. `RUNS` defaults to 7. For an original-versus-fast comparison, point `REFERENCE_ALIGNMENT` at the original demo's `src/lib/alignment` directory. This checks exact equality of Mel features, ONNX log probabilities and CTC paths on all clips. It also compares Mel features on silence, an impulse and two-sample inputs.

```sh
export REFERENCE_ALIGNMENT=/absolute/path/to/tinyaligner/src/lib/alignment
RUNS=15 BASELINE=1 npm run benchmark
RUNS=15 npm run benchmark
LOAD_WORKERS=2 RUNS=15 BASELINE=1 npm run benchmark
LOAD_WORKERS=2 RUNS=15 npm run benchmark
```

`LOAD_WORKERS` runs that many independent single-thread WASM sessions continuously during the benchmark. Their CPU time and memory count toward process-wide totals. `mainThreadCpuMs` is reported where supported by Node. For native CPU comparison, install `onnxruntime-node@1.26.0` in a scratch directory and set `NATIVE_ORT` to the absolute path to its `dist/index.js`. Use `THREADS=2` or `THREADS=4` in a fresh process for threading experiments. `BENCHMARK_MODEL` can point at an alternate model for a timing trial.

For browser WASM, open [the benchmark page](/benchmark) and select **Run benchmark**. To test threading, start with `BENCHMARK_ISOLATION=1 npm run dev` and open `/benchmark?threads=2` directly in a fresh tab. Check both `crossOriginIsolated` and `effectiveThreads` in the result. The demo defaults to one thread. The bundled runtime uses fixed-width SIMD even at one thread. Safari introduced this capability in 16.4, and the pinned package has no scalar binary. Changing the SIMD flag only changes feature detection. See [WebKit SIMD support](https://webkit.org/blog/13966/webkit-features-in-safari-16-4/) and [ONNX Runtime flags](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html).

### Apple M5 Max results

Node 25.2.1, ONNX Runtime 1.26.0, one inference thread, 15 timed iterations per clip. Values are median milliseconds, original → fast.

| Runtime | Audio | Preprocessing | Inference | CTC | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| WASM | 2.75 s | 4.03 → 1.90 | 5.67 → 6.04 | 0.05 → 0.06 | 9.84 → 8.05 |
| WASM | 10.99 s | 15.76 → 7.58 | 21.33 → 22.79 | 0.41 → 0.35 | 37.64 → 30.76 |
| WASM | 32.96 s | 47.10 → 23.01 | 63.59 → 67.16 | 3.06 → 2.71 | 114.01 → 93.44 |
| Native | 2.75 s | 4.07 → 2.01 | 1.39 → 1.73 | 0.06 → 0.07 | 5.58 → 3.80 |
| Native | 10.99 s | 15.86 → 8.02 | 5.57 → 6.87 | 0.41 → 0.38 | 21.81 → 15.20 |
| Native | 32.96 s | 47.58 → 22.50 | 16.41 → 18.15 | 3.08 → 2.69 | 66.59 → 43.73 |
| WASM + two active model workers | 2.75 s | 4.45 → 2.20 | 5.96 → 6.39 | 0.05 → 0.05 | 10.58 → 8.67 |
| WASM + two active model workers | 10.99 s | 17.27 → 8.53 | 23.40 → 24.24 | 0.47 → 0.39 | 40.93 → 32.93 |
| WASM + two active model workers | 32.96 s | 51.78 → 24.66 | 71.90 → 72.04 | 3.43 → 3.04 | 126.69 → 100.11 |

- Sparse Mel filtering skips zero coefficients in the frame loop. The FFT reuses its bit-reversal swaps and double-precision twiddle factors. The CTC backpointer table uses one byte per state per frame, about 1.24 MB for the longest clip versus 4.94 MB originally. Process-wide peak RSS varied across runs and includes the runtime and parity checks.
- Mel features, ONNX log probabilities and CTC paths matched exactly in the paired Node runs. The no-load single-thread WASM process used about 13% less CPU time across the full benchmark. With two competing WASM model sessions, the main thread used about 18% less CPU time. Both measurements include initialization and parity checks.
- In a separate WASM run, 1, 2 and 4 threads took 91.6, 64.2 and 55.7 ms on the longest clip. Total process CPU time over the benchmark rose from 2.37 s to 3.00 s and 4.82 s. One thread is the default for sustained concurrent use.
- Desktop Chrome 156 single-thread WASM went from 115.7 ms before the changes to 95.5 ms in the production preview for the longest clip. In isolated dev runs, Chrome took 88.0, 63.2 and 58.9 ms with 1, 2 and 4 threads. Playwright WebKit 27.2 took 83.7 and 55.8 ms with 1 and 2 threads.
- Dynamic INT8 quantization of the output projection reduced model bytes from 3.72 MB to 3.33 MB. The longest-clip WASM median was effectively flat at 91.6 ms FP32 versus 91.4 ms INT8. The tested dynamic quantizer left the GRU in floating point. The demo uses the pinned FP32 model.
- These figures cover repeated speech from one voice on one desktop. Older iPhone compatibility, thermal behavior, memory pressure and concurrent Phonon throughput need on-device measurement. Alignment still runs on the browser main thread.

### English-only projection experiment

The demo dictionary uses 39 phonemes plus silence and the CTC blank. The experimental model keeps those 41 output columns out of 258 and preserves the original FP32 weights for those columns and the full recurrent network. Removing unused columns changes the log-softmax normalization by a constant per frame in exact arithmetic. Every CTC path consumes one score per frame, so the constant cancels in path comparisons. Floating-point rounding can still affect near ties, so the benchmark checks exact CTC path parity.

```sh
python3 -m venv .venv
.venv/bin/pip install onnx==1.23.2
.venv/bin/python scripts/build-english-model.py
RUNS=15 npm run benchmark
BENCHMARK_VARIANT=english RUNS=15 npm run benchmark
```

The generator verifies the pinned source model and writes an alternate ONNX file and phone map under the ignored `static/models` directory. In the browser use `/benchmark?variant=english`. The variant compares each decoded path against the original model outside the timed iterations. Process-wide CPU and peak RSS include that parity work. For a second speech fixture, set `BENCHMARK_PCM` to its raw 16 kHz float32 file and `BENCHMARK_TRANSCRIPT` to its exact spoken text.

On the M5 Max, the 32.96 s fixture took **92.6 → 84.4 ms** total in single-thread Node WASM and **100.8 → 91.2 ms** with two additional WASM sessions active. Desktop Chrome took **89.9 → 82.5 ms** and Playwright WebKit took **84 → 72 ms**. The model went from 3.72 MB to 3.28 MB. CTC paths matched exactly on the three standard clip lengths in all three runtimes and on three lengths made from a second utterance and voice in Node. The English variant is opt-in for benchmarking.

Separate kernel prototypes kept the same GRU weights. A straightforward SIMD implementation took about 90 ms and packed weights with SIMD gate activations took about 59 ms, versus about 51 ms for the isolated ONNX Runtime GRU. The packed version matched CTC boundaries on the speech clips tested, but split-graph overhead and the slower recurrent implementation erased any benefit. The demo uses the ONNX Runtime GRU.

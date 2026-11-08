# James Facial Motion Studio

A local macOS sandbox for typing dialogue and watching the matching James v3
mesh speak it. The browser drives the skin, tongue, jaw, and eyes from the
batch-one ONNX graph. The audio player's clock selects the animation frame,
including after pause, seek, and replay.

## Vercel preview

Deploy from the `audio2face` directory with the Vite preset. The hosted
build prepares the James rig and serves the Pocket tokenizer module locally.
Vercel redirects model downloads to the pinned CDN and upstream URLs in
`vercel.json`. All speech synthesis and facial inference run in the browser.

The hosted project is **Liforma / james-audio2face**. Compare
[v4](https://james-audio2face.vercel.app/?model=compact-v4&sta=webgpu)
and [v3](https://james-audio2face.vercel.app/?model=compact-v3&sta=webgpu).
Both links select WebGPU; the model selector retains the dialogue when switching.

Tabs left open across a deployment can reference removed JavaScript chunks. The
app confirms a missing same-origin build asset, saves the dialogue and selections,
and refreshes once. It keeps ordinary network errors visible and avoids reload
loops. Run `node --test scripts/test-preload-recovery.mjs scripts/test-base64.mjs`
for the recovery and WebKit tokenizer checks. Hosted browser smoke tests are
`scripts/verify-tts-deployment.mjs` and `scripts/verify-preload-recovery.mjs`; set
`DEPLOYMENT_URL`, `PLAYWRIGHT_MODULE` and optionally `WEBKIT_EXECUTABLE`.

The hosted build plays a smoothed, precomputed DAWN idle loop during both
idle and speech. Live speech-driven head gestures use the local DAWN service
described below.

Playback shows timings from the most recent generation. **First TTS audio**
measures from Generate to the engine's first raw samples. **First playback ·
TTS + STA** measures to the first scheduled audio start with animation ready;
it includes scheduling lead but excludes unknown device output latency. Both
include cold model loading. **STA per block** averages feature extraction and
inference/readback time, excluding waits for TTS and model loading. Its audio
duration is the emitted crop, normally 500 ms from overlapping one-second
windows, with shorter boundary blocks. The real-time factor divides total
emitted audio duration by total inference time. Replaying retains the original
generation's timings. Run `node --test scripts/test-stream-timing.mjs` for the
first-audio observation checks.

```sh
vercel deploy --environment preview
```

## Run locally

Use Node.js 22.12 or later, and Chrome with
graphics acceleration and WebGPU enabled. Python with NumPy is needed once
to prepare the rig. Install `uv` for the local head motion runtime.

```sh
npm ci
mkdir -p .local

# Use the matching James v3 sample geometry and model data.
curl -fL https://media.githubusercontent.com/media/NVIDIA/Maya-ACE/release/2.0/sample_project/maya_geom/james_geom_v3.ma \
  -o .local/james_geom_v3.ma
curl -fL https://huggingface.co/nvidia/Audio2Face-3D-v3.0/resolve/b74132732fd9a9d29b237bec193ded64c9745e91/model_data_James.npz \
  -o .local/model_data_James.npz
python3 scripts/prepare_rig.py .local/james_geom_v3.ma .local/model_data_James.npz

# Download the v1 baseline and make it active.
compact_models=https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/geometry-compact-v1
curl -fL "$compact_models/model-v1.onnx" -o .local/model-v1.onnx
cp .local/model-v1.onnx .local/model.onnx
# Download the default browser voice and the two ONNX alternatives.
sh scripts/fetch_phonon.sh
sh scripts/fetch_onnx_tts.sh
sh scripts/setup_head_motion.sh
npm run dev
```

Open the URL printed by Vite. The dev server starts at port 5173 and preview
starts at 4173, moving to the next available port when needed.
Enter up to 500 characters
and select **Generate & play**. Use **Mouth close-up** to inspect lip contact,
or drag and scroll to change the camera. Use the audio controls to pause,
seek, or replay. The **Head motion** slider adjusts the pose during playback
or while paused. It defaults to 60%, with 0% holding the head steady and
100% restoring the original speech motion strength. James blinks and shifts
his head subtly while idle, including while playback is paused. Blinks follow
the viewer clock so they continue across clips.

**Animation model** switches between the original geometry model and
**Compact ARKit v1**, **v2**, **v3** and **v4**. All compact versions predict 52 coefficients,
with 39 active channels, and use the same James targets for the skin, tongue
and jaw. V2 improves lip closure at the same 2.25 MB model size.
V3 adds subtitle-selected training speech at the same size, improving general
lip sync with a small contact-F1 tradeoff on the older closure-focused test set.
V4 adds balanced Phonon/Supertonic multi-voice training and clean
Easton replay at the same 2.25 MB size. The initial v4 was validation-selected
seed 73, epoch 19; three-seed fresh confirmation improved mean lip-gap error by 5.4%.
See [validation results](docs/plans/compact-v4-validation-results.md).
The prior v4 revision is **gain-r1**, the subsequent teacher-labelled
loudness fine-tune (seed 37, epoch 15). It retains the same 2.25 MB download,
improving quiet/loud RMS by 3.4% and extra-quiet RMS by 7.3% versus a matched
fine-tuning control. Those are single-seed follow-up results, separate from the
earlier three-seed confirmation. See [gain results](docs/plans/compact-gain-robustness-results.md).
The preceding revision is **coverage-r2**, trained on broader distinct speech at
the same 2.25 MB size. New original-speech RMS improved 2.5% versus gain-r1,
with mixed closure results. The immutable gain-r1 predecessor is retained.
See [coverage results](docs/plans/compact-coverage-results.md).
The preceding revision is **scale-r3**, expanded-data seed 37, development-selected
step 2976. This exact export improves confirmation lip-gap RMS 6.52% versus
coverage-r2, with the same 2.25 MB download and GPU buffers. The 7.22% result
is the three-seed mean, not the score of this individual export. The initial
control-relative screen missed its target; publication is requested for iPhone
testing, not a claim of independent release confirmation. See
[data-scale results](docs/plans/compact-data-scale-results.md).
The current revision is **finetune-r4**, the ordinary fine-tuning control from
the feature-distillation study (seed 37, development-selected step 2976). It
improves fresh-test lip-gap RMS by 2.56% versus scale-r3, retaining the same
2.25 MB file and browser graph. No teacher-feature head is included.
Open `?model=compact-v4&sta=webgpu` to test v4, then select v3 to compare the
same dialogue. Switching reloads the page and retains the text, engine and view.
The demo opens with **Compact ARKit v4** selected in the model
dropdown on desktop, iPhone and iPad. Explicit `?model=` links still select
the requested version. **Animation
runtime** defaults to WebGPU. Its dropdown also offers an explicit CPU
option, using ONNX Runtime WASM; there is no automatic switch to CPU.
Browser playback checks cover both compact v3 and the v4 candidate on CPU.
On-device testing found CPU too slow for iPhone playback,
so memory improvements must be verified with WebGPU enabled.
Compact ARKit v2, v3 and v4 use a dedicated WebGPU runner with their ONNX weights.
It computes the log-mel features with an FFT and reuses GPU buffers for the
animation network. The other model and runtime combinations use ONNX Runtime.
Switching either setting reloads the page to release the previous model and
restores the dialogue, speech engine, camera preset and head motion setting.
For local use, fetch the models and their shared rig:

```sh
mkdir -p .local/arkit-student
arkit_assets=https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/arkit-student-v1
curl -fL "$arkit_assets/james-arkit-student.onnx" -o .local/arkit-student/james-arkit-student.onnx
curl -fL "$arkit_assets/james-blendshapes.bin" -o .local/arkit-student/james-blendshapes.bin
curl -fL https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/arkit-student-v2/james-arkit-student-v2.onnx \
  -o .local/arkit-student/james-arkit-student-v2.onnx
curl -fL https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/arkit-student-v3/james-arkit-student-v3.onnx \
  -o .local/arkit-student/james-arkit-student-v3.onnx
curl -fL https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/arkit-student-v4-finetune-r4/james-arkit-student-v4.onnx \
  -o .local/arkit-student/james-arkit-student-v4.onnx
```

The speech engines synthesize English locally in browser workers. Phonon
is the default and uses Easton. The ONNX alternatives run on WebAssembly
and use Supertonic 3 M1 or Pocket TTS Alba. The alternatives download about
387 MiB and 181 MiB respectively into the ignored `.local` directory.
[Supertonic 3 browser example](https://github.com/supertone-oss-archive/supertonic/tree/main/web)
· [Pocket TTS browser ONNX export](https://huggingface.co/spaces/KevinAHM/pocket-tts-web)

The downloaded model terms are stored next to the weights. The Pocket TTS
browser export uses CC BY-NC 4.0, and the Supertonic 3 weights use Open RAIL-M.

Phonon and Pocket TTS emit PCM chunks while generating. Supertonic 3 emits
a complete segment at a time. The same 16 kHz audio feeds playback and
animation. Playback starts when the first animation frames are ready, which
requires about half a second of audio plus inference time. Use **Stop
streaming** to cancel. Replay and seeking are available after live playback.

Head motion runs through DAWN’s pretrained pose network and HuBERT in a local
Python worker. The setup downloads about 1.3 GB into `.local/head-motion`.
The demo feeds it overlapping audio windows and blends the resulting angles
at each join. DAWN originally processes full clips, so this is a streaming
adaptation. The renderer uses the rotational channels with reduced strength
for James and drives the existing neck pivot. On page load, the same model
processes 12 seconds of silence. The local server caches a smoothed loop from
that prediction and the viewer blends it with the speech pose during playback.
[DAWN paper](https://arxiv.org/abs/2410.13726)

The demo reads the active export from `.local/model.onnx`. All four compact
exports use the same batch-one interface with `window`, `identity`, `emotion`,
`input_latents`, and `noise` as inputs and `prediction` and `output_latents`
as outputs. To audition another export, download it under its versioned
filename, copy it to `.local/model.onnx`, and reload the page.

The [staging artifact manifest](https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/geometry-compact-v1/manifest.json)
lists byte counts and SHA-256 checksums. Model weights go in the ignored
`.local` directory. The [model license](https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/geometry-compact-v1/LICENSE.pdf)
and [attribution notice](https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/geometry-compact-v1/NOTICE.txt)
are available alongside the exports.

| Export | Size | Lip-gap RMS ↓ | Closure F1 ↑ | `/m/` contacts ↑ |
| --- | ---: | ---: | ---: | ---: |
| `model-v1.onnx` | 72.16 MB | 0.0754 | 94.2% | 117/206 |
| `model-v2.onnx` | 72.28 MB | 0.0730 | 95.4% | 111/206 |
| `model-encoder-precision.onnx` | 73.83 MB | 0.0698 | 95.5% | 116/206 |
| `model-encoder-closure.onnx` | 73.92 MB | 0.0741 | 94.9% | 153/206 |

The table uses the same 27 held-out clips in Chrome. RMS and F1 measure
agreement with the full model. The contact count measures the rig's lip gap
during aligned `/m/` sounds. The closure model adds a smoothed detector
trained on aligned speech and reviewed scripted contrasts. On an additional
24 held-out contrasts, it made contact on 11 of 12 `/m/` sounds, compared
with 9 for the encoder model, with zero new closures on the 12 negative
clips. Median paired Chrome inference time was the same as the encoder
baseline across three clips and four repeats.

`npm run build` checks the frontend build. The local head motion API and
asset paths are also available under `npm run preview`.

The viewer uses the source topology and neutral geometry with simple preview
materials. The model assets, source rig files, and derived geometry stay in
the ignored `.local` directory. The app does not depend on another checkout
at runtime.

- [James v3 Maya sample scene and geometry](https://github.com/NVIDIA/Maya-ACE/tree/release/2.0/sample_project)
- [Model bundle and terms](https://huggingface.co/nvidia/Audio2Face-3D-v3.0)

## Compact model data-scale research

The unchanged compact network was tested with a larger independently synthesized
corpus and a matched training-budget grid. See [results and limitations](docs/plans/compact-data-scale-results.md)
and [reproduction instructions](scripts/research/data_scale/README.md). The study
initially left production unchanged; the subsequently requested scale-r3 release
uses its seed-37 model, retaining WebGPU as the browser default.

The follow-up [teacher-feature distillation experiment](docs/plans/compact-feature-distillation-results.md)
kept the same browser graph and found no meaningful gain beyond matched ordinary
fine-tuning. Its [reproduction instructions](scripts/research/feature_distillation/README.md)
and synchronized local research review are separate from the published demo;
the subsequent requested release uses the fine-tuned control as v4 finetune-r4.

The [targeted difficult-speech pilot](docs/plans/compact-targeted-speech-results.md)
compared context-enriched speech with matched broad speech and ordinary
fine-tuning. It found no meaningful extra lip-gap gain; production remains
finetune-r4. See its [reproduction instructions](scripts/research/targeted_speech/README.md).

The [dense lip-surface experiment](docs/plans/compact-lip-surface-results.md)
added raw-teacher 3D mouth supervision without increasing browser model size.
It produced a small gain beyond matched fine-tuning, with a closure trade-off,
and remains research-only. See its [reproduction instructions](scripts/research/lip_surface/README.md).

Teacher history/noise diagnostic: [protocol and results](docs/plans/teacher-history-compatibility-results.md), with [reproduction steps](scripts/research/teacher_history/README.md).

Jointly trained recurrent-state pilot: [matched quality, browser speed and memory results](docs/plans/compact-recurrent-state-results.md).

Current-v4 learnability: [eight-clip memorisation results](docs/plans/v4-memorisation-results.md).

Full-corpus loss balancing: [matched quality and browser results](docs/plans/compact-loss-balance-results.md).

Frozen mouth experts: [James compatibility results](docs/plans/mouth-expert-compatibility-results.md).

Training-only SyncNet: [matched quality, WebGPU speed and memory results](docs/plans/compact-syncnet-loss-results.md).

Full-corpus fit: [training/test measurements and frozen-readout diagnostic](docs/plans/compact-training-fit-results.md).

Closure-protected training: [matched quality, WebGPU speed and memory results](docs/plans/compact-closure-protected-results.md).

Jaw targets and intermediate teacher: [audit, matched distillation and browser results](docs/plans/compact-assistant-teacher-results.md).

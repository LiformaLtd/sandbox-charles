# James Facial Motion Studio

A local macOS sandbox for typing dialogue and watching the matching James v3
mesh speak it. The browser drives the skin, tongue, jaw, and eyes from the
batch-one ONNX graph. The audio player's clock selects the animation frame,
including after pause, seek, and replay.

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

Open [http://127.0.0.1:5178](http://127.0.0.1:5178/). Enter up to 500 characters
and select **Generate & play**. Use **Mouth close-up** to inspect lip contact,
or drag and scroll to change the camera. Use the audio controls to pause,
seek, or replay. The **Head motion** slider adjusts the pose during playback
or while paused. It defaults to 60%, with 0% holding the head steady and
100% restoring the original speech motion strength. James blinks and shifts
his head subtly while idle, including while playback is paused. Blinks follow
the viewer clock so they continue across clips.

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

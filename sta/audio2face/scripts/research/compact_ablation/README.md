# Compact v3 controlled ablations

These scripts preserve the architecture, optimization, data selection and evaluation
used in the 2026-11-07 comparison. They do not replace the demo's active models.
See `../../../docs/plans/learnings.md` for teacher label generation, rig fitting,
target definitions and dataset provenance.

The input corpora are research artifacts, not files bundled with this repository.
Set `A2F_RESEARCH_WORK` to their common directory. It must contain:

- `arkit-student/{manifest.json,labels/,rig-metrics.npz,channels.json}`
- `arkit-student-v2/{manifest.json,labels/}`
- `arkit-student-v3/{manifest.json,labels/,james-arkit-student-v3.onnx}`
- `arkit-student-v3/extra/{manifest.json,labels/}`

Each manifest row has `stem`, `split`, `text`, `audio` and optional `source`,
`kind`, `phones`. Audio paths must resolve to 16 kHz little-endian float32 PCM.
Each `labels/<stem>.npz` contains full 52-channel `weights` and nine raw teacher
`gaps`. Paths in historical manifests require relocation on another machine.
Do not silently skip missing labels. Keep the frozen train/dev/test assignments.

Outputs and feature caches go to `A2F_ABLATION_OUTPUT`, which defaults to
`audio2face/.local/compact-ablation`. For the data arm it also needs
`extra/{manifest.json,labels/}`, containing the new training-only corpus.

Training used Python 3.12.12, torch 2.4.1, NumPy 2.2.6 and CPU threads=6. ONNX 1.23.2
was imported from the teacher environment for export. Evaluation used a separate
Python 3.12 environment with ONNX Runtime 1.31.1 and NumPy 2.5.4. Export is opset 17.
Record dependency changes when using another environment.

Example commands, from this directory, with those inputs and dependencies prepared:

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/research-corpora
export A2F_ABLATION_OUTPUT=/absolute/path/to/output
python train.py --name control
python train.py --name width128 --width 128
python train.py --name width192 --width 192
python train.py --name context1500 --context 24000
python train.py --name contextwide --context 24000 --dilated-context
python train.py --name data --extra
python evaluate.py baseline "$A2F_RESEARCH_WORK/arkit-student-v3/james-arkit-student-v3.onnx"
python evaluate.py control "$A2F_ABLATION_OUTPUT/control.onnx"
python evaluate.py width128 "$A2F_ABLATION_OUTPUT/width128.onnx"
python evaluate.py width192 "$A2F_ABLATION_OUTPUT/width192.onnx"
python evaluate.py context1500 "$A2F_ABLATION_OUTPUT/context1500.onnx"
python evaluate.py contextwide "$A2F_ABLATION_OUTPUT/contextwide.onnx"
python evaluate.py data "$A2F_ABLATION_OUTPUT/data.onnx"
python summarize.py
```

`select-data.py` selects up to 64 unused subtitle prompts using training-only v3
errors; the available pool in this run yielded 61. Set `A2F_DICT_ROOT` to the sibling
dictionary checkout containing its lexicon and subtitle-analysis artifacts. The
script produces `extra-prompts.json`. Synthesis, alignment and teacher label
generation use the existing dictionary workflow described in the experiment
report; they are prerequisites to `train.py --extra`.

All arms start from released v3. They have the same 486 optimizer updates, losses,
validation selection score and legacy replay order. Width expansion copies
channels and divides fan-in weights by replication counts, preserving the
starting function apart from seeded 1e-5 weight noise. A separate data RNG prevents
network initialization size from changing the training-window order. The longer
window uses the same learned dilations and coefficients; it tests additional
real input context instead of zeros at the original window's boundaries. The `--dilated-context`
arm also changes block 5 from dilation 1 to 32, enlarging the feature receptive
field from 81 to 143 frames without adding parameters.

The current public WebGPU runner supports width 96 and 101 feature frames. Wider
or longer-window ONNX exports need the matching research runner modifications;
loading them into the unchanged demo is unsupported. Keep tensor dimensions,
shader dispatches, frontend, output interpolation and streaming crop in sync.

`summary.py` reports paired clip-bootstrap intervals (2,000 draws, seed 37).
Those intervals do not measure variability across independent training seeds.


## Browser and memory reproduction

The optional benchmark uses Playwright WebKit on macOS, with a libproc sampler.
Provide Playwright in a separate research environment; it is not a production app
dependency. `WEBKIT_EXECUTABLE` can pin a particular installed WebKit build;
`PLAYWRIGHT_MODULE` can point to its absolute `index.mjs` when needed.

```sh
python fixtures.py baseline control data width128 width192 context1500 contextwide
python prepare_browser.py
clang -O2 sample.c -o "$A2F_ABLATION_OUTPUT/sample"
# From the app directory, start the generated isolated harness:
node node_modules/vite/bin/vite.js --config "$A2F_ABLATION_OUTPUT/vite.config.mjs" --mode hosted
# In another terminal, use the actual next-free URL printed by Vite:
export BENCH_URL=http://127.0.0.1:5174
node scripts/research/compact_ablation/browser_bench.mjs baseline control data width128 width192 context1500 contextwide
python scripts/research/compact_ablation/summarize.py
```

Use an absolute output path. Do not run training concurrently with timing/memory
benchmarks. The fixture/parity page and full demo use separate browsers. The
sampler measures simultaneous WebContent/GPU footprints, not total machine memory.

The `.pt` files contain state dictionaries; reconstruct their width, window and
dilation settings from the JSON configuration before loading them. ONNX files
include the exported graph. Keep their hashes with each result.


## Longer matched-seed study

Use a fresh output directory for a new recipe. `study.py` freezes the training
and evaluation source, warms feature caches, runs matched width/seed pairs and
writes complete selected exports plus historical test results. All epochs keep
the same 48/16 new/replay batch composition. The default sweep is 12 epochs,
widths 96/128/192 and seeds 37/73/101; configure concurrency for your machine.

```sh
export A2F_ABLATION_OUTPUT=/absolute/path/to/fresh-study-output
python study.py --epochs 12 --seeds 37 73 101 --widths 96 128 192 --jobs 2
```

`--training-python` and `--evaluation-python` support separate pinned environments.
The optional `--onnx-site-packages` compatibility path imports NumPy/PyTorch from
the training environment before making the export dependency available. Prefer
fully configured environments when reconstructing the study on another machine.

The 48 original confirmation prompts are included as `confirmation-prompts.json`.
To recreate their audio and targets, set `A2F_DICT_ROOT` to the dictionary app
with its pinned Phonon/TinyAligner assets. Set `A2F_TEACHER_ROOT` to the compatible
offline teacher checkout containing `app/audio2face_v3` and
`models_cache/audio2face-v3/network.onnx`. `A2F_ASSET_ROOT` defaults to
`$A2F_RESEARCH_WORK/v3-model` and contains the James source fitting assets.

```sh
node --experimental-strip-types synthesize_confirmation.mjs
python label_confirmation.py
python evaluate_confirmation.py baseline
python evaluate_confirmation.py w96-s37
# Evaluate the other eight names in the same way after training finishes.
python summarize_study.py
```

`summarize_study.py` implements the recorded 12-epoch, three-seed cohort. Adjust
its explicitly declared cohort when designing a different study. It reports means,
standard deviations, individual runs and crossed paired bootstrap intervals that
resample seeds and whole clips together. Three seeds remain a limited estimate
of training variability. Representative checkpoints are selected by validation,
not confirmation scores.

After preparing the isolated browser harness and selecting representative models,
`build_review.py` creates six predetermined, synchronized mouth comparisons and
lip-gap plots. Open `/review.html` on that harness. All faces use the same recorded
audio, frame clock and seek position. The reference lane shows teacher-fitted
coefficients; it is not the full geometry renderer. A "Hold head steady" control
is for inspecting mouths and does not alter the demo's continuous idle policy.
The model files, tracks, audio and plots stay under the ignored output directory.

## Folding a replicated wider model

`fold_width.py` applies to the modulo-96 replication used by this recipe, with
width a multiple of 96. It sums input replicas and averages output replicas,
then exports the original 96-channel graph. It preserves the function exactly
only when replica features are identical; learned divergence can make it lossy.
It checks known graph structure and fixed buffers, but these checks do not replace
output parity and held-out evaluation. Width 128 is not supported by this helper.

```sh
python fold_width.py "$A2F_ABLATION_OUTPUT/w192-s37.onnx" "$A2F_ABLATION_OUTPUT/folded-s37.onnx"
python evaluate.py folded-s37 "$A2F_ABLATION_OUTPUT/folded-s37.onnx"
python evaluate_confirmation.py folded-s37
```

The follow-up study found differences below 9e-7 across 8.6 million coefficients
while restoring the original 2.25 MB file size. This is evidence for those
replicated, fine-tuned models; it is not a general pruning guarantee. See
[the follow-up report](../../../docs/plans/compact-v3-width-followup.md).

The benchmark pre-bundles `phonon-tts/worker.js`: discovering it lazily can make a
cold Vite dev server reload the page during preparation. Measured runs require
exactly one main-frame navigation. `BENCH_REPEATS` overrides the default two
fresh-browser repeats per model/clip; the follow-up uses three.

# Dense lip-surface supervision without a larger browser model

Read the [frozen pilot](../../../docs/plans/compact-lip-surface.md).
This is offline research starting from released v4 finetune-r4. Both arms export
only the existing compact waveform model; no new browser speech encoder,
phoneme timings, head, loss tensors or surface target files are shipped.

## What is supervised

The static James mouth region uses the earlier fitter's spatial bounds on
teacher neutral skin: `abs(x)<3`, `166.1<y<169`, `z>5` centimetres. It contains
2,267 vertices / 6,801 coordinates, including all 18 existing gap landmarks,
inner/outer lip geometry, lateral corners and nearby perioral skin. It is a
spatial region, not a manual vermilion segmentation. The local mask plot is
`surface-region.png`; the old frontal fit overlaps only part of this region.

Let teacher geometry be `S + D`, rig neutral `R`, and active rig basis `B`.
Prediction is `R + wB`, so target displacement relative to rig neutral is
`t = D + (S-R)`. For each axis cache `q = tBᵀ/V` and `s = mean(t²)`; fixed Gram
`G = BBᵀ/V`. Then `wGw - 2wq + s` equals mean squared dense coordinate error.
Sum axes for full 3D vertex MSE. Keeping `s` preserves error orthogonal to the
rig span. This is an exact linear-rig identity, not PCA or landmark sampling.

Targets are float32 sufficient statistics: 117 linear terms plus three squared
norms per frame. Do not quantize absolute head-height coordinates to float16.
The offline teacher wrapper projects its output before host transfer but keeps
the source graph, state, priming windows, seed and crop unchanged. Untouched
teacher parity and per-track nine-gap checks guard this optimization. Dense/
quadratic values, coordinate frames and CPU/MLX gradients are checked separately.

## Inputs and environment

First reproduce prior data-scale/feature/targeted studies or retain their inputs:
`.local/compact-scale-v1`, `.local/compact-feature-v1`, `.local/compact-targeted-v1`,
coverage/multivoice/gain/packed assets, rig/channels, historical replay, full
teacher and pinned TTS/aligner assets. The original teacher/rig model files live
in `A2F_RESEARCH_WORK/v3-model`. v4 is the immutable feature-study control export,
not a mutable local demo weight path. Its SHA is pinned in `common.py` and plan.

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/research-work
export A2F_SURFACE_OUTPUT="$PWD/.local/compact-surface-v1"
export A2F_TEACHER_ROOT=/absolute/path/to/ai-workflows/apple-silicon
export TRAIN_PYTHON=/absolute/path/to/torch-python
export EVAL_PYTHON=/absolute/path/to/onnx-python
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx-python/site-packages
export NODE22=/absolute/path/to/node22
```

Recorded versions: MLX0.31.2 float32 / TF32 disabled; PyTorch2.4.1 and NumPy2.2.6
for training/export; ONNX1.23.2, ORT1.31.1 and NumPy2.5.4 for evaluation;
Matplotlib3.10.8 for review plots; Node22.22.3 / desktop WebKit2388. The optional
site-packages path exposes ONNX/MLX to the original split training interpreter.
Use compatible environments and rerun numerical checks on a new setup.

`common.py` sets `A2F_SCALE_OUTPUT` to the new output before importing the old
Dataset loader. Its input map directories are read-only links to prior arrays.
Never point new output at an earlier study. Historical manifests contain
absolute audio paths: migrate/retain the full prior inventory for a new run.
The fresh-preparation audit includes the old normal checkout explicitly; change
that root before freezing a replication and report the new source/data hashes.
CMU text is unchanged from the pinned prior source; retain its license/notice.
No original human speech recordings are reused.

## Preparation and reference checks

```sh
"$EVAL_PYTHON" scripts/research/lip_surface/prepare_confirmation.py
"$EVAL_PYTHON" scripts/research/lip_surface/run_confirmation.py
```

The confirmation pipeline synthesizes and aligns 96 original recordings, then
independently teacher-labels 384 original/half/headroom-boost/quarter cases.
Unknown dictionary words must be fixed before preparation, not silently aligned.
No new texts may overlap any earlier training/dev/test, including targeting.
Four test presets are previously studied training-held-out voices, not new
speakers. Source, audio, teacher, synthesis and fitter provenance stay pinned.

Before surface labelling, verify the compact projection against untouched raw
geometry. The original setup used the first old-training row and wrote
`teacher-projection-parity.json` with this invocation:

```sh
"$EVAL_PYTHON" - <<'PY'
import sys,json
sys.path.insert(0,'scripts/research/lip_surface')
from common import *
from teacher import verify
inputs()
row=json.loads((SCALE/'datasets/old-train/manifest.json').read_text())[0]
freeze(OUT/'teacher-projection-parity.json',verify(row))
PY
"$EVAL_PYTHON" scripts/research/lip_surface/label_surface.py --cohort primary
"$EVAL_PYTHON" scripts/research/lip_surface/label_surface.py --cohort confirmation
"$EVAL_PYTHON" scripts/research/lip_surface/cache_surface.py
"$TRAIN_PYTHON" scripts/research/lip_surface/check.py
"$EVAL_PYTHON" -m unittest discover -s scripts/research/lip_surface -p 'test_*.py'
```

Raw-surface statistics are generated for all 3,840 primary training and 864
common-development cases; replay retains only its old animation losses.
The primary labeler verifies all regenerated gaps against the existing references,
plus exact sample indices and audio/recipe checksums. Waveform-hash keyed caches
may safely reuse identical gain-limited audio. Every distinct waveform gets its
own teacher inference, with state reset at the start of each utterance.

`cache_surface.py` can run alongside primary labelling: it waits for completed
atomic per-track files, writes aligned primary/dev maps and freezes hashes when
complete. Training never reads an incomplete map. Primary/confirmation labellers
may overlap because fresh waveforms are disjoint; their output locks are separate.
Retain failed setup logs, without mixing them into valid measurement results.

## Calibrate, freeze and train

```sh
"$TRAIN_PYTHON" scripts/research/lip_surface/calibrate.py
"$EVAL_PYTHON" scripts/research/lip_surface/freeze_training.py
"$TRAIN_PYTHON" scripts/research/lip_surface/train.py --arm control
"$TRAIN_PYTHON" scripts/research/lip_surface/train.py --arm surface
```

One frozen voice-balanced training minibatch calibrates the surface coefficient
so its initial backbone gradient norm is 25% of the base objective norm. It is
then fixed and ramped over 372 updates; control coefficient is zero. Both arms
share seed37, 3,720 updates, the same primary/replay batches, base losses,
normalization and optimizer schedule. There is no coefficient sweep.

Both use a shared NEW development selector combining gap RMS, half-weight dense
surface RMS and contact F1 on four common-dev voices; the Easton guard contributes
its unchanged gap/contact score as fifth voice. This is a joint surface-shape
experiment, so surface is primary; it is not an exact replay of the old gap-only
selector. All selections are development-only, with step0 allowed. Complete
model/optimizer/RNG states and sampler hashes are retained. `postprocess.py` can
wait for primary/map completion, calibrate, freeze and run both arms; it does not
score test data automatically.

## Fresh quality and browser verification

```sh
"$EVAL_PYTHON" scripts/research/lip_surface/build_review.py --freeze-only
"$EVAL_PYTHON" scripts/research/lip_surface/evaluate_experiment.py
"$EVAL_PYTHON" scripts/research/lip_surface/build_parity.py
"$TRAIN_PYTHON" scripts/research/lip_surface/build_review.py
```

Freeze visual cases and evaluator before scoring; lock both exports first.
Dense 3D RMS is `10*sqrt(mean(sum_axis MSE))` mm, with per-axis RMS reported too.
The fitted teacher coefficients provide a retargeting-residual reference, not a
strict globally optimal lower bound. Gap/contact/opening/closure/timing remain
independent checks. Historical/feature cohorts are inspected regression sets;
only the new confirmation is fresh. Keep failed acceptance screens unchanged.

Prepare/link `.local/rig.bin` and `rig.json` as well as model directories. Start
hosted-mode source Vite using the app README and take its printed free port.
Run benchmarks after all teacher, training and native reference activity stops:

```sh
export DIAGNOSTIC_URL=http://127.0.0.1:YOUR_PORT
export PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs
export WEBKIT_EXECUTABLE=/absolute/path/to/webkit/pw_run.sh
export A2F_PARITY_MODELS='{"v4":"/.local/compact-feature-v1/control/model.onnx","control":"/.local/compact-surface-v1/control/model.onnx","surface":"/.local/compact-surface-v1/surface/model.onnx"}'
"$NODE22" scripts/research/lip_surface/verify_browser.mjs
export SCALE_REVIEW_URL="$DIAGNOSTIC_URL/.local/compact-surface-v1/review.html"
"$NODE22" scripts/research/lip_surface/verify_review.mjs
cc -O2 scripts/research/compact_ablation/sample.c -o "$A2F_SURFACE_OUTPUT/sample-footprint"
export A2F_MEMORY_SAMPLER="$A2F_SURFACE_OUTPUT/sample-footprint"
"$NODE22" scripts/research/lip_surface/measure_memory.mjs
```

Preserve three fresh settled timing/parity runs. The four-lane review compares
fitted teacher/current v4/control/surface, with both gap and dense-error plots.
Normal-page memory checks retain idle animation and use prerecorded short/long/
repeated STA; TTS is excluded. Explicit GPU bytes are not total process memory.
20ms desktop WebContent+GPU samples can miss short peaks and exclude browser/
network processes; an emulated iPhone UA does not establish on-device safety.

```sh
"$EVAL_PYTHON" -m compileall -q scripts/research/lip_surface
"$NODE22" --check scripts/research/lip_surface/verify_browser.mjs
"$NODE22" --check scripts/research/lip_surface/verify_review.mjs
"$NODE22" --check scripts/research/lip_surface/measure_memory.mjs
git diff --check
```

Keep all model weights, generated speech, targets/maps, corpus/license/protocol
snapshots, checkpoints and raw reports in the ignored output directory. This
experiment does not upload a model or change production/CDN/Vercel defaults.

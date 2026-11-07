# Tiny causal temporal decoder pilot

[Protocol](../../../docs/plans/compact-temporal-decoder.md) ·
[Results](../../../docs/plans/compact-temporal-decoder-results.md)

This research-only experiment freezes published v4's audio frontend, encoder and
projection. A 12,967-parameter residual head consumes current plus seven prior
96-dimensional encoded motion features, with 16 GELU units and 39 corrections.
History is **features**, not corrected poses fed back autoregressively. It adds
no lookahead. The matched control repeats current features in all eight slots.
Its selected epoch-zero correction is zero, reproducing v4, despite completing
all training updates. This differs from earlier full-backbone fine-tune controls.

Use the training Python (torch/NumPy), evaluation Python (ONNX/ORT/teacher), Node
22, and native teacher checkout from preceding multivoice/coverage/state studies.
The prior ignored corpora and pinned browser/teacher assets are prerequisites.
For an isolated checkout, link immutable prior artifact directories under
`.local` or reconstruct them from earlier plans; install its own dependencies.
Do not share writable node_modules. Model weights, audio and generated caches
must not enter Git. All new artifacts belong to `.local/compact-temporal-v1`.

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/research-work
export A2F_TEACHER_ROOT=/absolute/path/to/compatible/apple-silicon
export TRAIN_PYTHON=/absolute/path/to/torch/python
export EVAL_PYTHON=/absolute/path/to/teacher-onnx/python
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx/site-packages
export A2F_TEMPORAL_OUTPUT="$PWD/.local/compact-temporal-v1"

"$EVAL_PYTHON" scripts/research/temporal_decoder/prepare.py
"$TRAIN_PYTHON" scripts/research/temporal_decoder/data.py
"$TRAIN_PYTHON" -m unittest discover -s scripts/research/temporal_decoder -p 'test_*.py'
"$EVAL_PYTHON" scripts/research/temporal_decoder/evaluate.py --freeze-only
for arm in control temporal; do
  "$TRAIN_PYTHON" scripts/research/temporal_decoder/train.py --arm "$arm"
done
"$TRAIN_PYTHON" scripts/research/temporal_decoder/audit.py
```

`data.py` reconstructs complete per-clip feature tracks from original overlapping
windows and checks their stitched target coefficients. It estimates normalization
from training frames only. Context is constructed across original block boundaries,
then sampled in 30-frame windows; it never crosses recording boundaries. New
primary and replay corpora, sampling/update budgets, initialization and losses
are identical between arms. All backbone weights remain frozen. Full optimizer,
scheduler, sampler and best-state checkpoints are retained; this pilot trainer
refuses to overwrite a completed export and does not implement partial resume.

Fresh confirmation generation uses the same pinned speech engines and labeler:

```sh
for engine in phonon supertonic; do
  A2F_MULTIVOICE_OUTPUT="$A2F_TEMPORAL_OUTPUT/confirmation/base" \
    node scripts/research/multivoice/synthesize.mjs "$engine"
done
"$EVAL_PYTHON" scripts/research/temporal_decoder/engine_views.py
for engine in phonon supertonic; do
  export A2F_MULTIVOICE_OUTPUT="$A2F_TEMPORAL_OUTPUT/confirmation/base-$engine"
  node --experimental-strip-types scripts/research/multivoice/align.mjs
  "$EVAL_PYTHON" scripts/research/multivoice/verify.py
  "$EVAL_PYTHON" scripts/research/multivoice/label.py
done
"$EVAL_PYTHON" scripts/research/temporal_decoder/merge_base.py
"$EVAL_PYTHON" scripts/research/temporal_decoder/confirmation_variants.py
"$EVAL_PYTHON" scripts/research/temporal_decoder/build_review.py --freeze-only
A2F_MULTIVOICE_OUTPUT="$A2F_TEMPORAL_OUTPUT/confirmation/variants" \
  "$EVAL_PYTHON" scripts/research/multivoice/label.py
"$EVAL_PYTHON" scripts/research/temporal_decoder/evaluate.py
"$TRAIN_PYTHON" scripts/research/temporal_decoder/build_review.py
"$EVAL_PYTHON" scripts/research/temporal_decoder/prepare_browser.py
"$EVAL_PYTHON" scripts/research/temporal_decoder/build_parity.py
```

Engine synthesis/owned views may run independently. Modified waveform gains get
new full-teacher labels. Exact prior transcript overlap is excluded, including
state/capacity confirmation. Voice presets are known, training-held-out presets,
not previously unseen speakers. In this run the evaluator was frozen after
training but before confirmation scoring; checkpoints were locked before scoring.

The exported model takes `pcm [1,16000]`, normalized `history [1,7,96]`, integer
`first [1]` and valid `count [1]`. It returns 30 coefficient rows and next history;
consume only `count` output rows. Count advances history only for emitted frames.
Skip all-padding blocks; zero history between utterances. Do not run ordinary
one-input compact inference on this export.

The browser prototype reuses the production FFT, convolution and projection,
adding a third WGSL pipeline, fixed history/output buffers and crop uniform.
`run(pcm,{first,count})` returns the original 60-frame layout; `run.reset()` zeros
history and `run.dispose()` destroys its device. Production code is untouched.
The ONNX exporter transposes Linear weights into generated MatMul initializers;
`prepare_browser.py` resolves graph node inputs and transposes them back, then
locks exact names and source hashes. The first failed packing setup is retained
under `browser-setup1-failed`; no trained weights were changed to fix packing.

Run Vite in hosted mode on its actual next-free port. It needs immutable rig,
blendshape and idle-loop assets. Export `DIAGNOSTIC_URL`, `PLAYWRIGHT_MODULE` and
`WEBKIT_EXECUTABLE` for the pinned local WebKit. Set:

```sh
export A2F_ACTIVATION_OUTPUT="$A2F_TEMPORAL_OUTPUT/browser/parity"
export A2F_ACTIVATION_ASSET_PATH=/.local/compact-temporal-v1/browser/parity/
export A2F_PARITY_MODELS='{"v4":"/.local/compact-coverage-v2/coverage/runs/control-replay-s37/model.onnx","control":"/.local/compact-temporal-v1/control/model.onnx","temporal":"/.local/compact-temporal-v1/temporal/model.onnx"}'
export A2F_PARITY_RUNNERS='{"control":"/.local/compact-temporal-v1/browser/control-runner.js","temporal":"/.local/compact-temporal-v1/browser/temporal-runner.js"}'
node scripts/research/temporal_decoder/verify_browser.mjs
```

Preserve three `activation-parity.json` runs as `run-N.json`. Cases include eight
fresh engine/gain recordings, both complete historical packed tracks and a
reset repeat of the first clip. The reset repeat must be byte-identical. The
unit tests also cover causality in feature time, cross-block history, short tails,
zero-frame blocks and history-independent control. Use `measure_startup.mjs` for
three fresh-browser cached-local-asset startup trials. Set `TEMPORAL_REVIEW_URL`
to the generated review URL for `verify_review.mjs` (sixteen synchronized cases).

For macOS renderer/GPU physical footprint at 20 ms sampling:

```sh
mkdir -p "$A2F_TEMPORAL_OUTPUT/memory"
clang -O2 scripts/research/compact_ablation/sample.c \
  -o "$A2F_TEMPORAL_OUTPUT/memory/sample"
export A2F_MEMORY_SAMPLER="$A2F_TEMPORAL_OUTPUT/memory/sample"
node scripts/research/temporal_decoder/measure_memory.mjs
```

This launches fresh WebKit per model, identifies its new WebContent/GPU PIDs,
keeps the normal James page animating, and runs short/long/repeated prerecorded
STA streams. It sums simultaneous physical footprints, not independent per-PID
maxima. Sampling can miss sub-20-ms peaks; no TTS session is prepared. These are
not full TTS+STA pipeline or physical-iPhone peak-memory measurements. Small
cross-run differences are noise, not evidence that extra weights save memory.

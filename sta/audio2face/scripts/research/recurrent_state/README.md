# Jointly trained compact recurrent-state pilot

[Protocol](../../../docs/plans/compact-recurrent-state.md) ·
[Results](../../../docs/plans/compact-recurrent-state-results.md)

This is isolated research, not a production selector or automatic promotion.
Native evaluation uses Python with MLX 0.31.2, NumPy 2.5.4, ONNX 1.23.2 and ORT
1.31.1. Export/review use PyTorch 2.4.1, NumPy 2.2.6, ONNX 1.16.2 and Matplotlib
3.10.8. Browser tools use Node 22+, Playwright 1.65.0 and desktop WebKit revision
2388. Install app dependencies with `npm ci` in this dedicated worktree.

Set `A2F_RESEARCH_WORK` to the existing research asset workspace and
`A2F_TEACHER_ROOT` to the original `ai-workflows/apple-silicon` checkout. Set
`EVAL_PYTHON`, `TRAIN_PYTHON`, `NODE22` to their executables. The pinned original
teacher, James rig metrics, scale training/dev frontend caches, source corpus
manifests/labels/audio, feature-control v4 export, earlier confirmation corpora,
packed diagnostics and pinned multivoice synthesis assets are prerequisites.
See prior data-scale and teacher-history instructions to reconstruct them.
`common.py` states the exact expected layout; immutable prior artifact directories
can be linked into this worktree's `.local`. Do not share writable node_modules.
All new output is `.local/compact-recurrent-v1`, ignored by Git.

From the app directory, with a fresh output directory:

```sh
"$EVAL_PYTHON" scripts/research/recurrent_state/data.py
"$EVAL_PYTHON" scripts/research/recurrent_state/prepare_confirmation.py
"$EVAL_PYTHON" scripts/research/recurrent_state/test_model.py
"$EVAL_PYTHON" scripts/research/recurrent_state/freeze_training.py
"$EVAL_PYTHON" scripts/research/recurrent_state/train.py --arm control
"$EVAL_PYTHON" scripts/research/recurrent_state/train.py --arm recurrent
"$EVAL_PYTHON" scripts/research/recurrent_state/run_confirmation.py
"$TRAIN_PYTHON" scripts/research/recurrent_state/export.py
"$EVAL_PYTHON" scripts/research/recurrent_state/native_reference.py
"$EVAL_PYTHON" scripts/research/recurrent_state/check_export.py
"$EVAL_PYTHON" scripts/research/recurrent_state/evaluate_experiment.py
"$EVAL_PYTHON" scripts/research/recurrent_state/test_runtime.py
"$EVAL_PYTHON" scripts/research/recurrent_state/prepare_browser.py
"$EVAL_PYTHON" scripts/research/recurrent_state/build_parity.py
"$TRAIN_PYTHON" scripts/research/recurrent_state/build_review.py
```

Training and confirmation synthesis can proceed independently after the freeze;
finish both before export/scoring. Train all backbone parameters plus the cell/
head; never substitute frozen features. Checkpoint selection uses complete ordered
development utterances only. The shared index/reset schedule, data/source hashes,
full per-voice progress, best/last weights, optimizer and stream state are retained.
Automatic interrupted-run resume is not implemented. Do not rerun setup or change
frozen files in an existing experiment; preserve the directory before a new run.

The exported prototype takes `pcm [1,16000]`, `state [1,32]`, integer `first [1]`
and `count [1]`; it returns `[1,30,52]` rows and next state. Only the first `count`
rows are emitted. The first input crop can begin at row 30; ordinary crops start
at 15. Skip empty blocks and reset between utterances. `inference.py` implements
this contract. The WebGPU runner returns the existing 60-frame layout and accepts
`run(pcm,{first,count})`; its `.reset()` and `.dispose()` manage state/device.
Do not pass this export to the ordinary one-input compact ONNX inference path.

Run `npm run dev -- --mode hosted`, using Vite's actual next-free port. James
rig/blendshape assets and the hosted idle loop must be available. Set:

```sh
export DIAGNOSTIC_URL=http://127.0.0.1:ACTUAL_PORT
export PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs
export WEBKIT_EXECUTABLE=/absolute/path/to/webkit/pw_run.sh
export A2F_RECURRENT_OUTPUT="$PWD/.local/compact-recurrent-v1"
export A2F_ACTIVATION_OUTPUT="$A2F_RECURRENT_OUTPUT/browser/parity"
export A2F_ACTIVATION_ASSET_PATH=/.local/compact-recurrent-v1/browser/parity/
export A2F_PARITY_MODELS='{"v4":"/.local/compact-feature-v1/control/model.onnx","control":"/.local/compact-recurrent-v1/control/model.onnx","recurrent":"/.local/compact-recurrent-v1/recurrent/model.onnx"}'
export A2F_PARITY_RUNNERS='{"control":"/.local/compact-recurrent-v1/browser/control-runner.js","recurrent":"/.local/compact-recurrent-v1/browser/recurrent-runner.js"}'
node scripts/research/recurrent_state/verify_browser.mjs
export RECURRENT_REVIEW_URL="$DIAGNOSTIC_URL/.local/compact-recurrent-v1/review.html"
node scripts/research/recurrent_state/verify_review.mjs
```

Preserve three parity reports as `browser/parity/run-N.json`; each run verifies
all eleven fixtures plus an exact reset repeat against native references.
`measure_startup.mjs` records three fresh-browser cached-local startup trials.
For physical footprint (macOS only):

```sh
mkdir -p "$A2F_RECURRENT_OUTPUT/memory"
clang -O2 scripts/research/compact_ablation/sample.c -o "$A2F_RECURRENT_OUTPUT/memory/sample"
export A2F_MEMORY_SAMPLER="$A2F_RECURRENT_OUTPUT/memory/sample"
node scripts/research/recurrent_state/measure_memory.mjs
node scripts/research/recurrent_state/measure_startup.mjs
```

Retain each memory trial's v4/control/recurrent JSON files under `memory/run-N/`
before another trial overwrites the current outputs. Run three trials without
other GPU benchmarks concurrently. These sum simultaneous WebContent+GPU physical
footprints, sampled at 20 ms, from loaded James idle through model startup and
accelerated short/long/three-repeat STA. A test-only route hook passes coefficient
poses into the normal James renderer and verifies advancing idle draw calls.
There is no TTS session and no physical iPhone measurement. Read scope limits in
the results before using the numbers for a memory budget.

Finally `analyze.py` computes supplementary transcript-cluster bootstrap, dense
surface regression, export-tensor equality and three-run browser summaries. It
requires all three `run-N` reports and was written after primary scoring; it
never selects checkpoints. Generated models/audio/arrays/plots stay out of Git.

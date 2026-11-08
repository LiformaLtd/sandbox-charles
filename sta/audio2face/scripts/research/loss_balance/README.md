# Full-corpus coefficient-loss balance pilot

[Protocol](../../../docs/plans/compact-loss-balance.md) ·
[Results](../../../docs/plans/compact-loss-balance-results.md)

Research only. Current v4 finetune-r4 is the pinned initialization; both arms use
all existing scale primary and historical replay training data. The only intended
change is normalized coefficient SmoothL1 weight 1→4; all other loss terms and
training/sampling/selection settings are identical. This factor is a predeclared
hypothesis, not a selected optimum or a sweep. No automatic publication.

Prerequisites are the pinned existing research corpora/features/teacher/rig and
browser synthesis assets from the data-scale, feature-distillation, surface and
recurrent studies. Immutable artifact directories may be symlinked under `.local`
in this dedicated worktree; never share writable node_modules. Set
`A2F_RESEARCH_WORK` to the workspace containing `arkit-student` rig metrics and
`A2F_TEACHER_ROOT` to the native teacher's `ai-workflows/apple-silicon` checkout.
`common.py` documents the expected `.local` layout. All new artifacts go to
`.local/compact-loss-v1`, ignored by Git. Prior source corpora/model files are not
bundled here. Install app dependencies with `npm ci`.

Use Python with MLX0.31.2, NumPy2.5.4, ONNX1.23.2 and ORT1.31.1 for training/
evaluation (`EVAL_PYTHON`). The trainer's final export needs PyTorch2.4.1;
`A2F_TORCH_SITE_PACKAGES` can point to its Python3.12 site-packages if it is not in
the evaluation environment. Review/plots use NumPy2.2.6 / Matplotlib3.10.8
(`TRAIN_PYTHON`). Node22+ is `NODE22` for synthesis/alignment.

With a fresh output directory:

```sh
"$EVAL_PYTHON" scripts/research/loss_balance/test_loss.py
"$EVAL_PYTHON" scripts/research/loss_balance/prepare_confirmation.py
"$EVAL_PYTHON" scripts/research/loss_balance/freeze_training.py
"$EVAL_PYTHON" scripts/research/loss_balance/train.py --arm control
"$EVAL_PYTHON" scripts/research/loss_balance/train.py --arm balanced
"$EVAL_PYTHON" scripts/research/loss_balance/run_confirmation.py
"$EVAL_PYTHON" scripts/research/loss_balance/audit.py
"$EVAL_PYTHON" scripts/research/loss_balance/evaluate_experiment.py
"$EVAL_PYTHON" scripts/research/loss_balance/build_parity.py
"$EVAL_PYTHON" scripts/research/loss_balance/check_exports.py
"$TRAIN_PYTHON" scripts/research/loss_balance/build_review.py
```

Training and synthesis can run independently after freeze. Complete both before
scoring. The trainer preserves model/optimizer/best/RNG/sampler state, verifies
frozen hashes and supports same-source checkpoint resume. Do not overwrite a
completed export. If export dependencies fail, install/configure those dependencies
and resume without editing frozen training sources. All 3,720 updates are shared
in budget/sampling, including six windows per eight voices and sixteen replay.
The same development selector retains epoch-zero. Logged gradient norms and
clipping frequencies expose the scaling effect of changing the loss coefficient.

The preparation text audit includes prior recurrent confirmation and all known
previous planned corpora; unsupported dictionary words are rejected before any
training/synthesis. Speech synthesis uses pinned browser engines/voices/assets.
Every changed gain waveform receives its own full-teacher label. Scoring checks
audio hashes against labels and locks checkpoints before any confirmation score.
Primary evaluation is fresh gap RMS plus contact/event guards; prior cohorts are
regression evidence. No test-set weight tuning follows a failed screen.

For browser checks, run `npm run dev -- --mode hosted` and use its actual next-free
port. Rig/blendshape assets and the hosted idle track must be available. Set:

```sh
export DIAGNOSTIC_URL=http://127.0.0.1:ACTUAL_PORT
export PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs
export WEBKIT_EXECUTABLE=/absolute/path/to/webkit/pw_run.sh
export A2F_LOSS_OUTPUT="$PWD/.local/compact-loss-v1"
export A2F_ACTIVATION_OUTPUT="$A2F_LOSS_OUTPUT/browser/parity"
export A2F_ACTIVATION_ASSET_PATH=/.local/compact-loss-v1/browser/parity/
export A2F_PARITY_MODELS='{"v4":"/.local/compact-feature-v1/control/model.onnx","control":"/.local/compact-loss-v1/control/model.onnx","balanced":"/.local/compact-loss-v1/balanced/model.onnx"}'
export A2F_PARITY_RUNNERS='{}'
node scripts/research/loss_balance/verify_browser.mjs
export LOSS_REVIEW_URL="$DIAGNOSTIC_URL/.local/compact-loss-v1/review.html"
node scripts/research/loss_balance/verify_review.mjs
```

All three use the unchanged production WebGPU runner and one-input waveform model.
Retain three parity JSONs as `browser/parity/run-N.json`. Verify reset repeats are
byte-identical. Browser tools use Playwright1.65.0, WebKit revision2388 /27.2.

```sh
mkdir -p "$A2F_LOSS_OUTPUT/memory"
clang -O2 scripts/research/compact_ablation/sample.c -o "$A2F_LOSS_OUTPUT/memory/sample"
export A2F_MEMORY_SAMPLER="$A2F_LOSS_OUTPUT/memory/sample"
node scripts/research/loss_balance/measure_memory.mjs
node scripts/research/loss_balance/measure_startup.mjs
```

Retain each memory trial's three JSONs under `memory/run-N/` before another trial.
Use three trials with no other GPU benchmarks/teacher training concurrently.
Physical footprint sums simultaneous WebContent+GPU samples at20ms, starting
from loaded animated James idle, through session creation, short/long/three-repeat
accelerated prerecorded STA. A test-only pose hook applies coefficients and checks
that rendering continues. TTS, initial rig loading and browser-parent/network
processes are excluded. This is not a physical-iPhone full-pipeline memory test.

After all three runs, `analyze.py` adds supplementary transcript-cluster bootstrap,
prior-cohort dense-surface metrics, constant-tensor equality and browser summaries.
It never selects checkpoints. Generated models/audio/arrays/plots stay out of Git.

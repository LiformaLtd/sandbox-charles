# Training-only articulation state

Frozen [protocol](../../../docs/plans/compact-articulation-state.md).
This study retains the 96-channel model and browser runner, adding only offline
supervision. It requires the preceding coverage, capacity, multivoice and replay
artifacts plus the original teacher checkout. Weights, recordings, labels,
optimizer checkpoints and source locks remain outside Git.

Run from the app directory with the pinned environments from the earlier studies:

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/research-work
export A2F_TEACHER_ROOT=/absolute/path/to/compatible/apple-silicon
export TRAIN_PYTHON=/absolute/path/to/torch/python
export EVAL_PYTHON=/absolute/path/to/teacher-and-onnx/python
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx/site-packages
export A2F_STATE_OUTPUT="$PWD/.local/compact-state-v1"

"$EVAL_PYTHON" scripts/research/articulation_state/prepare.py
"$EVAL_PYTHON" scripts/research/articulation_state/evaluate.py --freeze-only
"$TRAIN_PYTHON" -m unittest discover -s scripts/research/articulation_state -p 'test_*.py'
for arm in control state; do
  A2F_MULTIVOICE_OUTPUT="$A2F_STATE_OUTPUT/$arm" \
    "$TRAIN_PYTHON" scripts/research/articulation_state/train.py \
    --arm "$arm" --width 96 --seed 37 --epochs 24 --steps-per-epoch 31 \
    --lr .00012 --threads 4 --clean-replay \
    --baseline "$PWD/.local/compact-coverage-v2/coverage/runs/control-replay-s37/model.onnx"
done
"$TRAIN_PYTHON" scripts/research/articulation_state/audit.py
```

The two arms can train concurrently. `audit.py` freezes selected exports before
confirmation scoring, verifies identical data/replay/sampler/initial metrics,
checks exported graph topology/size, and records state-class support and head
weight movement. Both arms instantiate identical training heads; only the state
arm adds its head's loss. No head or state classifier is exported.

Generate confirmation with pinned browser-engine implementations. Either run the
engines sequentially or independently with their separate synthesis locks:

```sh
for engine in phonon supertonic; do
  A2F_MULTIVOICE_OUTPUT="$A2F_STATE_OUTPUT/confirmation/base" \
    node scripts/research/multivoice/synthesize.mjs "$engine"
done
"$EVAL_PYTHON" scripts/research/articulation_state/engine_views.py
for engine in phonon supertonic; do
  export A2F_MULTIVOICE_OUTPUT="$A2F_STATE_OUTPUT/confirmation/base-$engine"
  node --experimental-strip-types scripts/research/multivoice/align.mjs
  "$EVAL_PYTHON" scripts/research/multivoice/verify.py
  "$EVAL_PYTHON" scripts/research/multivoice/label.py
done
"$EVAL_PYTHON" scripts/research/articulation_state/merge_base.py
"$EVAL_PYTHON" scripts/research/articulation_state/confirmation_variants.py
A2F_MULTIVOICE_OUTPUT="$A2F_STATE_OUTPUT/confirmation/variants" \
  "$EVAL_PYTHON" scripts/research/multivoice/label.py
"$EVAL_PYTHON" scripts/research/articulation_state/evaluate.py
"$EVAL_PYTHON" scripts/research/articulation_state/build_parity.py
```

Engine views own mutable alignment and teacher source files; base output merges
only complete labels from the identical teacher recipe. Every altered waveform
level receives new teacher output, rather than copying original-level targets.
The fresh corpus excludes prior normalized transcripts, including capacity's
confirmation. Its known training-held-out voice presets are not new voices.

For parity/speed, use the same Vite-hosted source and standalone WebKit helper
as the capacity study. Set `DIAGNOSTIC_URL` to its actual next-free local port,
`A2F_ACTIVATION_OUTPUT` to `$A2F_STATE_OUTPUT/browser/parity`,
`A2F_ACTIVATION_ASSET_PATH=/.local/compact-state-v1/browser/parity/`,
`A2F_PARITY_MODELS` to the JSON map of `control` and `state` to their selected
ONNX source URIs, plus `PLAYWRIGHT_MODULE` and `WEBKIT_EXECUTABLE`. Run
`node scripts/research/multivoice/verify_activation_regression.mjs` in three
fresh contexts, preserving each `activation-parity.json` as `run-N.json`.
The generator uses eight fresh gain/engine cases and both previously recorded
complete “packed” tracks from `.local/packed-diagnostic`. It verifies selected
model hashes and locks audio, native references and shader source hashes.
Use the unchanged production runner; no prototype or runtime head is needed.

Do not infer iPhone or full-page peak memory from explicit model buffers. Warm
model block timing excludes TTS, rendering and startup. Leave published v4 and
idle animation unchanged during the experiment.


Completed pilot: [results](../../../docs/plans/compact-articulation-state-results.md).
For the post-hoc fitting-versus-prediction decomposition, run
`"$EVAL_PYTHON" scripts/research/articulation_state/fitting_diagnostic.py` after
scoring. This diagnostic never selects checkpoints. To rebuild the sixteen-case
mouth comparison, run `build_review.py --freeze-only` after preparing variants
and before scoring, then `build_review.py` after evaluation (use the training
Python environment with matplotlib for rendering). Run `verify_review.mjs` with
`STATE_REVIEW_URL` set to its generated local review URL and the same Playwright/
WebKit environment variables used for parity. It verifies audio and exact seek
frames. Source and export locks remain in the ignored artifact directory.

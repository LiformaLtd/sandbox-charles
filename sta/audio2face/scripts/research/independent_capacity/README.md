# Independent channel capacity

[Protocol](../../../docs/plans/compact-independent-capacity.md) ·
[Results](../../../docs/plans/compact-independent-capacity-results.md)

Run from the app directory using the pinned training/teacher/Node environments
from the multivoice and coverage studies. This follow-up requires their frozen
corpora and compatible original teacher checkout; it is not a standalone model
or a new deployment. Preserve ignored `.local` artifacts to reproduce exact
checkpoints. Downloads, audio and optimizer state must not enter Git.

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/research-work
export A2F_TEACHER_ROOT=/absolute/path/to/compatible/apple-silicon
export TRAIN_PYTHON=/absolute/path/to/torch/python
export EVAL_PYTHON=/absolute/path/to/teacher-and-onnx/python
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx/site-packages
export A2F_CAPACITY_OUTPUT="$PWD/.local/compact-capacity-v1"

"$EVAL_PYTHON" scripts/research/independent_capacity/prepare.py
"$EVAL_PYTHON" scripts/research/independent_capacity/evaluate.py --freeze-only
"$TRAIN_PYTHON" -m unittest discover \
  -s scripts/research/independent_capacity -p 'test_*.py'

for width in 96 128; do
  if [ "$width" = 96 ]; then arm=control96; else arm=independent128; fi
  A2F_MULTIVOICE_OUTPUT="$A2F_CAPACITY_OUTPUT/$arm" \
    "$TRAIN_PYTHON" scripts/research/independent_capacity/train.py \
    --arm control --width "$width" --seed 37 --epochs 24 \
    --steps-per-epoch 31 --lr .00012 --threads 4 --clean-replay \
    --baseline "$PWD/.local/compact-coverage-v2/coverage/runs/control-replay-s37/model.onnx"
done
"$TRAIN_PYTHON" scripts/research/independent_capacity/audit.py
```

Synthesize the new confirmation only, using the planned 96 recordings. Both
engine processes use separate locks. After both finish, align, verify teacher
instrumentation and label the complete base; then label every changed waveform
level. Original-level labels are linked only when the waveform hashes match.

```sh
export A2F_MULTIVOICE_OUTPUT="$A2F_CAPACITY_OUTPUT/confirmation/base"
node scripts/research/multivoice/synthesize.mjs phonon
node scripts/research/multivoice/synthesize.mjs supertonic
node --experimental-strip-types scripts/research/multivoice/align.mjs
"$EVAL_PYTHON" scripts/research/multivoice/verify.py
"$EVAL_PYTHON" scripts/research/multivoice/label.py
"$EVAL_PYTHON" scripts/research/independent_capacity/confirmation_variants.py
A2F_MULTIVOICE_OUTPUT="$A2F_CAPACITY_OUTPUT/confirmation/variants" \
  "$EVAL_PYTHON" scripts/research/multivoice/label.py
"$EVAL_PYTHON" scripts/research/independent_capacity/evaluate.py
"$EVAL_PYTHON" scripts/research/independent_capacity/prepare_browser.py
```

`prepare.py` freezes initialization SHA, source hashes, normalized-text exclusion,
training manifest and confirmation plan before training. `audit.py` checks both
full training budgets, shared source/data/replay, equal sampler state, initial
function metrics and learned extra features, then locks selected model hashes.
The evaluator separately reports fresh recordings and previously inspected
historical/coverage cohorts. Do not tune on confirmation after scoring.

For browser checks, serve the app with Vite and `VITE_HOSTED_DEMO=true` so `.local`
fixtures and the isolated prototype can be imported. Prepare a `parity-cases.json`
in an ignored fixture directory using the format consumed by
`../multivoice/verify_activation_regression.mjs`: each case has `name`, audio URI,
`reference` keyed by model name and native float32 track reference URIs. Native
references must use `compact_ablation/metrics.py:infer` on complete audio tracks.
Set `DIAGNOSTIC_URL`, `A2F_ACTIVATION_OUTPUT`, `A2F_ACTIVATION_ASSET_PATH`,
`A2F_PARITY_MODELS` (JSON name-to-ONNX URI map), `A2F_PARITY_RUNNERS` (JSON map
assigning the width-128 model to the generated prototype module URI),
`PLAYWRIGHT_MODULE`, and `WEBKIT_EXECUTABLE`, then run that helper three times.
Keep each JSON result rather than overwriting the only evidence. The results
report documents the ten cases and timing aggregation used in this study.

The generated runner patches hidden width, convolution dispatch and projection
stride/loop sizes. It reuses the production frontend and stable convolution
shader, records source hashes, and refuses mismatched frozen outputs. Public
runtime files and idle animation remain untouched. Model buffer instrumentation
counts explicit allocations only, not a physical iPhone or whole-page peak.

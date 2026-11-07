# Distinct speech coverage follow-up

Protocol: [compact-coverage.md](../../../docs/plans/compact-coverage.md).
This is an offline, same-architecture data experiment. Published v4 gain-r1
stays unchanged. It requires the previous multivoice/gain-study corpora, frozen
assets/sources, clean replay corpora and native teacher environment. The related
dictionary project's `pair-frequencies.json` is statistical input only; no raw
subtitle lines or fitted dictionary animation tracks are reused.

Use the pinned Node/training/teacher environments from the earlier studies.
Run from the app directory. Artifacts remain outside Git.

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/earlier-research-work
export A2F_TEACHER_ROOT=/absolute/path/to/compatible/apple-silicon
export TRAIN_PYTHON=/absolute/path/to/training/python
export EVAL_PYTHON=/absolute/path/to/teacher-and-evaluation/python
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx/site-packages
export A2F_COVERAGE_OUTPUT="$PWD/.local/compact-coverage-v2"

"$EVAL_PYTHON" scripts/research/coverage/prepare.py
"$EVAL_PYTHON" scripts/research/coverage/engine_views.py
"$EVAL_PYTHON" scripts/research/coverage/evaluate.py --freeze-only

# These two synthesis processes may run concurrently. Each has its own lock.
A2F_MULTIVOICE_OUTPUT="$A2F_COVERAGE_OUTPUT/base" \
  node scripts/research/multivoice/synthesize.mjs phonon
A2F_MULTIVOICE_OUTPUT="$A2F_COVERAGE_OUTPUT/base" \
  node scripts/research/multivoice/synthesize.mjs supertonic

# Each engine view can be processed as soon as its synthesis is complete.
for engine in phonon supertonic; do
  export A2F_MULTIVOICE_OUTPUT="$A2F_COVERAGE_OUTPUT/base-$engine"
  node --experimental-strip-types scripts/research/multivoice/align.mjs
  "$EVAL_PYTHON" scripts/research/multivoice/verify.py
  "$EVAL_PYTHON" scripts/research/multivoice/label.py
done
"$EVAL_PYTHON" scripts/research/coverage/merge_base.py
"$EVAL_PYTHON" scripts/research/coverage/prepare_variants.py
"$EVAL_PYTHON" scripts/research/coverage/build_review.py --freeze-only
A2F_MULTIVOICE_OUTPUT="$A2F_COVERAGE_OUTPUT/variants" \
  "$EVAL_PYTHON" scripts/research/multivoice/label.py
"$EVAL_PYTHON" scripts/research/coverage/audit.py

for arm in control coverage; do
  A2F_MULTIVOICE_OUTPUT="$A2F_COVERAGE_OUTPUT/$arm" \
    "$TRAIN_PYTHON" scripts/research/multivoice/train.py \
      --arm control --clean-replay --seed 37 --epochs 24 --steps-per-epoch 31 \
      --baseline "$PWD/.local/compact-gain-v2/augmented/runs/control-replay-s37/model.onnx"
done
# Freeze selected model hashes/epochs before the first evaluation.
"$EVAL_PYTHON" scripts/research/coverage/select_checkpoint.py
"$EVAL_PYTHON" scripts/research/coverage/audit.py
"$EVAL_PYTHON" scripts/research/coverage/evaluate.py
"$TRAIN_PYTHON" scripts/research/coverage/build_review.py
"$EVAL_PYTHON" -m unittest discover -s scripts/research/coverage -p 'test_*.py'
```

`prepare.py` verifies dictionary support and normalized transcript isolation
against prior datasets, pins source/asset/model hashes and checks text phone
coverage with and without historical replay. The new training assignment uses
one Phonon and one opposite-gender Supertonic preset per text, rotating equally
across the eight existing training voices. Both arms have 256 base / 768
gain-conditioned primary training recordings; only their transcript distribution
changes. Historical replay and development inputs are common.

Per-engine views own mutable alignment/teacher outputs, avoiding concurrent
writes through a frozen source symlink. `merge_base.py` validates full label
tracks and a common recipe. `prepare_variants.py` uses exactly the prior gain
policy and reuses labels only for byte-identical waveforms. It copies unchanged
phoneme timestamps but regenerates geometry for every changed waveform.

`audit.py` checks eight-way voice balance, transcript splits, audio/label hashes,
shared development data and teacher provenance. After training it also checks
identical initial scores/configuration, equal update budgets and byte-for-byte
equal development tensor values. Model exports and raw audio stay ignored.

The evaluator reports fresh-transcript results and historical regression results
separately. Test voice presets were previously studied, but were held out from
training; do not label them newly discovered voices. Quarter gain is absent
from training/development. The old v4 confirmation set is not used for selection.

The scientific viewer has four synchronized lanes and twelve cases fixed by
voice/category/condition before test scoring. Serve the app in hosted-mode Vite
and open `/.local/compact-coverage-v2/review.html`. Set `COVERAGE_REVIEW_URL` to
that page and run `verify_review.mjs` to check all cases' audio and exact seek
frames. The stationary-head review mode is independent of the normal demo's
continuous idle animation. Do not benchmark browser memory during training.

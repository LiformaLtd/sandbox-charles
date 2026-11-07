# Teacher-labelled gain follow-up

Protocol: [compact-gain-robustness.md](../../../docs/plans/compact-gain-robustness.md).
Run from the Audio2Face app directory. Use the same pinned training and teacher
environments as the multivoice study. No demo code or model weights are replaced.

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/earlier-research-work
export A2F_TEACHER_ROOT=/absolute/path/to/compatible/apple-silicon
export TRAIN_PYTHON=/absolute/path/to/training/python
export EVAL_PYTHON=/absolute/path/to/teacher-and-evaluation/python
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx/site-packages
export A2F_GAIN_OUTPUT="$PWD/.local/compact-gain-v2"

"$EVAL_PYTHON" scripts/research/gain/prepare.py
"$EVAL_PYTHON" scripts/research/gain/evaluate.py --freeze-only
A2F_MULTIVOICE_OUTPUT="$A2F_GAIN_OUTPUT" \
  "$EVAL_PYTHON" scripts/research/multivoice/label.py
"$EVAL_PYTHON" scripts/research/gain/audit.py
"$EVAL_PYTHON" scripts/research/gain/build_review.py --freeze-only

# Wait for all teacher labels before running either arm. Both use v4, the
# same development cohort, clean replay and exactly 744 optimizer updates.
for arm in control augmented; do
  A2F_MULTIVOICE_OUTPUT="$A2F_GAIN_OUTPUT/$arm" \
    "$TRAIN_PYTHON" scripts/research/multivoice/train.py \
      --arm control --clean-replay --seed 37 --epochs 24 --steps-per-epoch 31 \
      --baseline "$PWD/.local/compact-v4-validation/v4-candidate.onnx"
done
"$EVAL_PYTHON" scripts/research/gain/evaluate.py
"$TRAIN_PYTHON" scripts/research/gain/build_review.py
"$EVAL_PYTHON" -m unittest discover -s scripts/research/gain -p 'test_*.py'
```

`prepare.py` verifies the v4 hash, freezes source/protocol hashes and transformed
waveforms, and prepares the two arm manifests. It refuses to overwrite changed
frozen artifacts. All labels, including original-level recordings, are generated
under the same current labeler recipe. The original pilot's labels remain intact.
Half-gain and headroom-capped boost are training/development conditions. Quarter
gain appears only in held-out evaluation. Scalar multiplication preserves the
audio timeline, so the existing phone timestamps identify the same phonemes;
actual closure geometry is regenerated, never shifted or copied from another gain.

The two CPU training processes can run concurrently in their separate arm output
directories. Do not benchmark browser memory/speed during training. Freeze the
evaluation source before training, evaluate only after both exports are complete,
and keep the prior fresh v4 confirmation set out of checkpoint selection.

The evaluator writes per-condition/per-engine/per-voice metrics and corrected
closure counts, opening errors, coefficients and velocity diagnostics. RMS is in
millimetres in the report, while raw teacher arrays retain source centimetres.
It checks the predeclared first-seed gates. This previously inspected pilot cohort
is not fresh release confirmation, even though the gain conditions have new labels.

Setup-v1 artifacts, if present, record an aborted cache-provenance check. Setup v2
uses newly generated labels throughout; no cache guard is weakened to resume v1.

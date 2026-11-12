# Full-corpus fit and frozen-readout diagnostic

[Results](../../../docs/plans/compact-training-fit-results.md).
No production changes or model publication. Outputs remain ignored under
`.local/compact-fit-v1`. Run in the dedicated research worktree with the immutable
artifact links used by the preceding SyncNet/data-scale experiments.

Set `A2F_RESEARCH_WORK` to the directory containing `arkit-student` rig metrics
and `arkit-student-v3`/`extra` replay manifests and labels. Use the prior
`A2F_TEACHER_ROOT` environment setting for existing research module imports; this
diagnostic does not run the teacher or synthesize new speech. `EVAL_PYTHON` needs
MLX0.31.2, NumPy2.5.4, ONNX1.23.2 and ORT1.31.1. Required local model/data paths
and pinned model hashes are in `common.py` and the prior research guides.

```sh
"$EVAL_PYTHON" scripts/research/fit_diagnostic/measure.py
"$EVAL_PYTHON" scripts/research/fit_diagnostic/head_refit.py
"$EVAL_PYTHON" scripts/research/fit_diagnostic/head_evaluate.py
"$EVAL_PYTHON" scripts/research/fit_diagnostic/audit.py
"$EVAL_PYTHON" scripts/research/fit_diagnostic/summarize.py
```

Use a fresh output directory for different data, code, models or hypotheses.
Frozen JSON rejects overwrite with different contents. The head diagnostic uses
the ordinary fine-tuned control, not the SyncNet arm. It fixes the encoder,
eight temporal blocks, interpolation, normalization and waveform frontend.
Only the final affine readout changes:3,783 existing parameter values.

## Measurement

Score every primary and replay training frame, not a sampled minibatch. Primary
has3,840 labelled clips/751,498 frames; replay453+96 clips/99,188 frames. Report
all three gain conditions and the comparable original-gain subset separately.
The original-gain primary subset has1,280 clips; this condition count is distinct
from the number of pre-gain independent recordings in the data-scale study.
Development864+29 clips is separate; the previous96-recording/384-gain test set
remains separate and is now inspected regression evidence for the new head.

Predictions use the cached log-mel frontend, MLX backbone and exact streaming
crop positions/masks, reassembled into complete60fps clips. All cached target
coefficients/gaps must match original labels frame for frame. Native waveform
ORT parity is checked on29 fixed boundary/voice cases per model. The adapter's
RMS, contact F1 and bilabial hits match the existing evaluator on768 clips.

Recompute and verify each feature-cache input fingerprint, including source,
manifest, actual audio hashes and complete label bytes. Historical replay uses
older schemas: extra manifests lack individual audio hashes, and some labels
lack the newer audio hash field. Their original frozen cache fingerprints bind
these bytes to the training arrays; missing fields are not treated as proof of
provenance. Any declared audio hash must match. The two earlier schema-adapter
failures and sources are preserved in ignored attempt folders.

Primary macro RMS pools squared gap error per voice, then averages voice RMS,
matching existing evaluation. Contact F1 and0–100ms anticipatory event scoring
also match. Training/development/test voices and text distributions differ;
the difference is not a controlled estimate of pure generalization error.
Teacher-fitted James reference discrepancy is measured on the same population.
It is not a strict lip-gap optimum, and RMS values must not be subtracted as
independent error components.

## One readout experiment

The predeclared objective is unclipped normalized coefficient squared error.
Primary frames have75% total weight, equally divided among eight voices and
uniform over frames within each voice. Replay frames have25% total weight.
This is an explicit frame-weighted diagnostic, not an exact recreation of the
stochastic mixed-loss batch normalization or a matched SGD-budget study.

Accumulate feature Gram/cross statistics in float64, center/standardize features,
and solve by eigendecomposition with fixed relative cutoff1e-9. All96 feature
directions survive; the normal-equation residual is1.32e-15. No ridge sweep,
checkpoint selection or development/test tuning is performed. Export float32
weights; playback still clamps coefficients to[0,1]. Thus the analytic optimum
is for the stated unclipped coefficient objective, not clipped lip-gap/contact
quality or all possible readouts of the frozen features.

The audit verifies that only the two output-layer initializers change, the
operator graph stays identical, file size remains2,251,827 bytes and native
parity stays below5e-5. No new browser speed/memory measurement is claimed.
The contact regression rejects this head for release; no new independent
confirmation or deployment follows a failed diagnostic.

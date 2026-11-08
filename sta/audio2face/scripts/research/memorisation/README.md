# Current-v4 tiny-set memorisation

[Protocol](../../../docs/plans/v4-memorisation.md) ·
[Results](../../../docs/plans/v4-memorisation-results.md)

This is a training-only diagnostic. It must not replace deployed weights.
An earlier four-clip test already succeeded; this rechecks current v4 with eight
hash-selected original-level training clips, one per training voice. Both arms
share every input and optimizer setting, differing only in training objective.

Use the existing research asset layout from data-scale/feature-distillation.
Set `A2F_RESEARCH_WORK` to the research workspace containing `arkit-student` rig
metrics. The app's `.local/compact-scale-v1` must contain original old-train
manifests/audio/labels, immutable feature caches and `sources/model.py`;
`.local/compact-feature-v1/control/model.onnx` must be the pinned current v4.
Read-only symlinks to prior artifacts are acceptable in an isolated worktree.
These prior corpora/models are not bundled in Git. All new artifacts are written
to `.local/v4-memorisation-v1`. Start in a fresh output directory.

Evaluation/training environment: Python, MLX 0.31.2, NumPy 2.5.4, ONNX 1.23.2,
ORT 1.31.1. Export/plots use PyTorch 2.4.1, NumPy 2.2.6, ONNX 1.16.2,
Matplotlib 3.10.8. Set `EVAL_PYTHON` and `TRAIN_PYTHON` to those executables.

```sh
"$EVAL_PYTHON" scripts/research/memorisation/prepare.py
"$EVAL_PYTHON" scripts/research/memorisation/train.py --arm mixed
"$EVAL_PYTHON" scripts/research/memorisation/train.py --arm fit-mse
"$TRAIN_PYTHON" scripts/research/memorisation/export.py
"$EVAL_PYTHON" scripts/research/memorisation/check.py
"$EVAL_PYTHON" scripts/research/memorisation/frontend_check.py
"$TRAIN_PYTHON" scripts/research/memorisation/plot.py
```

`prepare.py` freezes selection, full-batch maps, recipe and primary sources before
training. Both arms use full-batch chronological window maps, no augmentation,
3,000 updates and the same epoch-zero-inclusive checkpoint schedule. Best/last
weights and optimizer snapshots are saved; automatic resume is not implemented.
Keep interrupted outputs before starting again. Frozen artifacts refuse changed
writes. Selection is fitted-target gap RMS on the memorised clips; this is never
held-out model selection. Retain final and selected metrics, and the subset's own
reference instead of borrowing the 96-clip reference from another experiment.

`check.py` audits all source/audio/label/map hashes, tensor equality, parameter/
file sizes, complete streaming output and aggregate native-versus-training RMS.
It records the strict 5e-5 coefficient-parity flag without hiding a failure.
`frontend_check.py` is supplementary analysis of the observed tolerance miss;
it does not change any model weights or primary metrics. Models/audio/checkpoints,
plots and generated arrays stay out of Git. No node dependencies, server, browser
model switch, CDN upload or deployment is needed for this diagnostic.

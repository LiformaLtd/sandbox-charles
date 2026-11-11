# Intermediate animation teacher / compact feature distillation (LIFORMA-104)

Offline experiment only. No published model, Player file, mapping, URL or CDN asset is changed. The branch is based on the latest research `finetune-r4` snapshot, not the older sandbox primary checkout. Sources and input caches are read-only dependencies, never writable symlinks to another checkout.

Set `A2F_INTERMEDIATE_INPUT` to the existing feature-distillation application root containing `.local/compact-scale-v1` and `.local/compact-feature-v1`, and `A2F_RESEARCH_WORK` to the historical `arkit-student*` corpus root. Set `A2F_INTERMEDIATE_OUTPUT` to a new isolated result directory (default `.local/intermediate-teacher-v1`). Training interpreter: Python 3.12 / MLX / PyTorch 2.4.1 / NumPy 2.2.6 / ONNX; the completed protocol records versions and hashes. Do not install into another environment without recording a new experiment.

```sh
python scripts/research/intermediate_teacher/test_experiment.py
python scripts/research/intermediate_teacher/freeze.py
python scripts/research/intermediate_teacher/train.py --arm mentor
# Only if development-data mentor gate passes:
python scripts/research/intermediate_teacher/train.py --arm control --seed 37
python scripts/research/intermediate_teacher/train.py --arm distill --seed 37
# Repeat matched pair for seeds 73 and 101.
```

Mentor width 192, independent random extra channels, original 96-channel path preserved initially; 7,440 updates. Not replicated-channel Net2Wider folding. Student width 96, 3,720 updates per arm/seed. Same sampler, initialization, animation objective, AdamW schedule and train-only 96-to-192 hint heads in both arms. Distillation alone ramps a valid-frame cosine loss on frozen mentor layers 5 and 7 to 0.1 over 372 updates. The teacher features are computed on the same windows including replay examples. Mean/scale, context and coefficient interface are unchanged. Only the compact student is exported; hint heads and mentor never ship. This adapts EA's idea to our temporal architecture, not an exact reimplementation of its FC feature loss.

Development-only checkpoint selection includes epoch zero and a historical development guard. A failed mentor development gate stops distillation, as predeclared. Previously inspected held-out recordings can screen/regression-test this hypothesis, but are not a fresh independent confirmation. No model promotion is permitted without fresh confirmation and visual review.

Completed and interrupted runs refuse overwrite. Checkpoints retain model/optimizer state and RNG metadata for recovery; automatic resume is deliberately not implemented in this first experiment. A changed source or recipe requires a new output root, not altered historical result files.


## Evaluation and reports

After every compact pair is complete, run `evaluate.py` to lock all six development-selected exports and verify paired sampler/initialization hashes before reading held-out scores. `--mentor-only` separately screens the already selected teacher without selecting compact checkpoints. The original-condition feature/scale/historical cohorts remain distinct; no fresh-confirmation claim is made.

`check_export.py <run-name>` compares native MLX predictions with the waveform ONNX export on three training clips. `check_gradients.py` verifies the hint objective reaches the compact encoder while the frozen teacher stays unchanged. `build_report.py` takes `A2F_REPORT_OUTPUT` and writes a Markdown report, fixed metadata-selected lip-gap plots and provenance copies. Extra diagnostic scripts added after the initial training freeze have their own source hashes in their reports; they do not alter the frozen training recipe.

The export check uses an explicit NumPy reduction for the fixed channel map. Its first mentor-only diagnostic used NumPy/Accelerate matmul, which raised floating-point flags despite finite parity results; that initial report/source are retained separately, and the explicit-reduction v2 check is the accepted parity result. No warning suppression is used.

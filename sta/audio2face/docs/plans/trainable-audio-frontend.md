# Fixed versus learned mel-filter bank

LIFORMA-105, 11 November2026. User authorizes the experiment while retaining current model, 2.5D mapping and Player. Isolated `research/trainable-audio-frontend`, based on a00ff68; read-only finetune-r4 baseline ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da.

Question: can task-specific adaptation of the spectral frontend improve articulation beyond matched ordinary fine-tuning? This first test learns the existing mel matrix from identical initialization. It is not a full raw-waveform encoder trial.

Create fixed FFT-power caches from the exact original audio windows and verify reconstructed log-mel features against all existing input arrays. Preserve data splits, balanced voices, old replay, losses, predictor topology, Fourier/Hann and normalization. Learn the64x257 matrix directly, with nonnegative projection after each update and frontend learning rate.01 times the predictor rate. Fixed controls use stop-gradient and zero frontend decay; separate gradient clipping/optimizer state for predictor and frontend. Three seeds37/73/101,3,720 updates, same batches, cosine schedule and dev-only checkpoint selection with epoch zero eligible.

Freeze source/data/protected-file hashes before training; lock selected exports before scoring. Existing feature/scale/history cohorts and original/half/boost/quarter conditions are inspected regression screens, not fresh confirmation. Predeclared primary mean gap gain>=3% versus paired controls, no mean contact/opening/timed-closure-total loss, no engine mean regression, no mean gain-condition regression, historical mean gap within2% of baseline. A failed screen stops; no test-driven retuning. Positive results still require new independent recordings, real browser checks and visual review.

Expected browser topology/file size stay unchanged because the runtime already reads the mel tensor. Learned parameter count grows by16,448; nonzero bandwidth and frontend CPU cost must be measured rather than inferred from artifact size. No files/defaults/models outside this experiment are edited or deployed. See `scripts/research/trainable_frontend/README.md` for reproduction and environment inputs.


Before quality evaluation, a partial initial pilot was stopped: forward clamping gave zero gradients at absent bank coefficients. No held-out scores were read and no model was promoted. V2 uses projected direct weights and adds a zero-support activation test. Its independent source/protocol freeze and runs use `.local/trainable-frontend-v2`; verified power caches remain read-only in v1 with the original builder/provenance preserved. The aborted v1 source/logs remain available, not counted as experimental evidence.


## Completed corrected experiment

The corrected three-seed pilot did not pass: primary original-level mean gap error is0.2369% worse than paired fixed controls, contact F1 delta−0.000666. Scale/historical mean gap reductions are−0.3275%/−0.0613%. Gain-condition RMS also regresses; primary well-timed closure totals change+2/+7/−3, insufficient to override the other failed guards. These are existing held-out cohorts, not fresh independent confirmation. No candidate is promoted.

All six exports preserve2,251,827-byte original ONNX topology. Learned bank adds16,448 trainable values within an already existing tensor; predictor remains331,047 parameters. Positive weights increase491→15,596/15,846/15,798. A warm Node25 CPU frontend microbenchmark shows~0.541ms fixed versus1.861–1.871ms learned. This is not whole-browser/GPU/mobile timing.

Eight tests, six export parity checks (max7.75e-7), paired initialization/sampler checks and final integrity audit pass:27 protected primary files,15 frozen sources,6 original data roots and6 cached-power roots unchanged. No Player/mapping/active-model/default/deployment edits. No fresh confirmation or full-avatar preference study was pursued after the failed screen. Code stays isolated and uncommitted; checkpoints/evaluations/logs and aborted-v1 provenance remain ignored. User report, gap/bank figures, code/evidence archive and integrity/benchmark JSON are in the chat outputs directory.

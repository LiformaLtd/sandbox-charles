# Closure/timing-protected optimizer-step pilot

[Protocol](../../../docs/plans/compact-closure-protected.md).
All outputs stay ignored under `.local/compact-protected-v1`. No production
changes, new browser architecture or automatic publication. The initialization
is the prior ordinary fine-tuning control, not the SyncNet or readout-refit model.

Use the immutable dataset/model/rig links from the preceding studies. Set
`A2F_RESEARCH_WORK`, `A2F_TEACHER_ROOT`, `EVAL_PYTHON`, `TRAIN_PYTHON` and `NODE22`
as in the data-scale/SyncNet reproduction guides. MLX0.31.2 / NumPy2.5.4 /
ONNX1.23.2 / ORT1.31.1 are used for training/evaluation. Export needs Torch2.4.1;
`A2F_TORCH_SITE_PACKAGES` may point to its Python3.12 environment. Plots use
NumPy2.2.6 / Matplotlib3.10.8. No SyncNet, differentiable renderer or PyTorch3D
is needed. App dependencies come from the lockfile via `npm ci`.

## Frozen preparation

```sh
"$EVAL_PYTHON" scripts/research/closure_protected/preflight.py
"$EVAL_PYTHON" scripts/research/closure_protected/test_protection.py
"$EVAL_PYTHON" scripts/research/closure_protected/check_steps.py
"$EVAL_PYTHON" scripts/research/closure_protected/prepare_confirmation.py
"$EVAL_PYTHON" scripts/research/closure_protected/freeze_training.py
```

Conflict checks use64 fixed training batches at the baseline, resetting the
optimizer for each probe; they are not a model-training trajectory. Tests verify
known projection solutions, random feasibility/nonexpansion and real nonlinear
guard acceptance. Freeze rejects failed preflights or changed sources. Different
data, algorithms or settings require a new output directory/protocol.

The original loss terms/weights stay unchanged. The treatment projects the actual
AdamW parameter displacement onto two guard halfspaces, then backtracks on the
same batch if either weighted contact/timing loss increases. It is not standard
PCGrad and does not mathematically guarantee held-out F1 or event counts. Adam
moments/counters retain the attempted update even when accepted motion is reduced
or skipped. See `protection.py`, `step.py` and the protocol for precise tolerances.

## Matched training and independent quality checks

```sh
"$EVAL_PYTHON" scripts/research/closure_protected/train.py --arm control
"$EVAL_PYTHON" scripts/research/closure_protected/train.py --arm protected
"$EVAL_PYTHON" scripts/research/closure_protected/run_confirmation.py
"$EVAL_PYTHON" scripts/research/closure_protected/audit.py
"$EVAL_PYTHON" scripts/research/closure_protected/build_review.py --freeze-only
"$EVAL_PYTHON" scripts/research/closure_protected/evaluate_experiment.py
"$EVAL_PYTHON" scripts/research/closure_protected/measure_training.py
"$EVAL_PYTHON" scripts/research/closure_protected/summarize_fit.py
"$EVAL_PYTHON" scripts/research/closure_protected/build_parity.py
"$EVAL_PYTHON" scripts/research/closure_protected/check_exports.py
"$TRAIN_PYTHON" scripts/research/closure_protected/build_review.py
```

Synthesis/labeling may run independently after corpus freeze. Finish training
and labeling before scoring; checkpoint locks precede fresh confirmation metrics.
Both arms receive the same3,720 scheduled updates, batches, replay, optimizer
schedule and development selector, with initialization eligible. Accepted
parameter movement and wall time are not matched. Checkpoints atomically save
optimizer/model/best/RNG and guard history; resume verifies the sampler prefix.
Do not edit training sources after freeze.

Full primary/replay training fit uses the earlier exact-frame diagnostic and
cache/audio/label fingerprint checks, with native waveform parity. `fit/` keeps
its separate model/data/source locks. Public-v4 and initialization fit reports
are reused only when their model hashes match. Original-gain primary1280 clips
and all-primary3840 cases stay distinct. Replay is measured separately.

The fresh cohort is24 disjoint texts×four test presets:96 original recordings
plus separately teacher-labelled gain variants (384 cases). All previous planned
texts, including SyncNet confirmation, are excluded. The original closure/event
scorer retains anticipatory timing. Report false closures, missed contacts and
conditional timing errors alongside F1 and per-phone hits. `summary.json` is the
fresh-test screen; `fit-summary.json` additionally requires the predeclared
training-error gain. Existing cohorts are regression evidence.

## Browser speed, memory and review

Run the hosted-mode Vite server using its next free port, and set `DIAGNOSTIC_URL`
to that actual URL. Set `PLAYWRIGHT_MODULE` to Playwright1.65.0's `index.mjs`, and
`WEBKIT_EXECUTABLE` to installed WebKit2388 `pw_run.sh`. Finish all GPU training
and labeling before measurements.

```sh
mkdir -p .local/compact-protected-v1/memory
clang -O2 scripts/research/compact_ablation/sample.c -o .local/compact-protected-v1/memory/sample
node scripts/research/closure_protected/run_browser_checks.mjs
"$EVAL_PYTHON" scripts/research/closure_protected/analyze.py
```

The wrapper saves three parity trials, checks reset identity, saves three isolated
physical-footprint trials per model, measures startup and verifies16 synchronized
review cases. V4, control and protected use the same production WebGPU runner.
The legacy measurement environment names `A2F_LOSS_OUTPUT`/`LOSS_REVIEW_URL` are
set by the wrapper. Native parity covers both-engine/gain clips, short tails and
the earlier packed-phrase regression.

Physical footprint sums new WebContent+GPU processes every20ms with animated
James, session startup, short/long and repeated accelerated prerecorded STA.
It excludes TTS, initial rig loading, network and browser-parent processes. It
is a desktop WebKit test, not a full-pipeline physical-iPhone memory measurement.
Idle rendering must continue. No browser model defaults, CDN assets or Vercel
configuration are changed by the study.

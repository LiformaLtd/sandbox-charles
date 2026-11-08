# Jaw audit and assistant-teacher pilot

[Design](../../../docs/plans/compact-assistant-teacher.md). No app defaults,
production model, deployment or runtime dependencies change. Generated evidence
is ignored under `.local/compact-assistant-v1`.

This study depends on the preceding immutable data-scale corpus/feature caches,
published v4 initialization, NVIDIA original model, James blendshape assets and
Maya ARKit scene. Set `A2F_RESEARCH_WORK` to their work root and `A2F_TEACHER_ROOT`
to `liforma-audio2face-v3/ai-workflows/apple-silicon` within it. The original
geometry model is pinned by its existing runner; use the data-scale reproduction
guide for the corpus and cache construction. Do not point this study at
independently regenerated or modified caches without a new protocol directory.

`EVAL_PYTHON` uses MLX0.31.2, NumPy2.5.4, ONNX1.23.2 and ORT1.31.1.
`A2F_TORCH_SITE_PACKAGES` can expose Torch2.4.1 Python3.12 for ONNX export.
`TRAIN_PYTHON` is the NumPy2.2.6/Matplotlib3.10.8 environment for review plots.
Node22.12+ and locked app dependencies are used for browser verification.

```sh
"$EVAL_PYTHON" scripts/research/assistant_teacher/audit_targets.py
"$EVAL_PYTHON" scripts/research/assistant_teacher/audit_surfaces.py
"$EVAL_PYTHON" scripts/research/assistant_teacher/preflight.py
"$EVAL_PYTHON" scripts/research/assistant_teacher/train_assistant.py --arm assistant
"$EVAL_PYTHON" scripts/research/assistant_teacher/train_assistant.py --arm control
"$EVAL_PYTHON" scripts/research/assistant_teacher/train_assistant.py --arm distilled
"$EVAL_PYTHON" scripts/research/assistant_teacher/evaluate_pilot.py
"$EVAL_PYTHON" scripts/research/assistant_teacher/build_parity.py
"$EVAL_PYTHON" scripts/research/assistant_teacher/check_exports.py
"$TRAIN_PYTHON" scripts/research/assistant_teacher/build_review.py
```

Run the arms sequentially. The tiny arms refuse to start if the assistant fails
its predeclared development gap/F1 gate. Widening parity, complementary-column
gradient asymmetry and initial waveform export parity are checked before source
and data locks. Same seed/batches and update counts make the two tiny arms
matched; offline assistant training is additional compute. Training is short
and deliberately has no resume path; a crash requires a new archived trajectory,
never continuing with an altered sampler. Best checkpoints and full development
histories are retained. Test/regression scores never select checkpoints.

The jaw audit preserves all old labels and saves sufficient statistics instead
of full teacher geometry. It checks cached-label parity, recomputes native Maya
jaw deltas, and tests a fixed soft constraint on a small train/dev sample.
The assistant experiment uses the original labels regardless of audit results.
Existing confirmation sets have already been inspected and are regression
evidence, not a fresh independent test or release justification.

Run a hosted-mode Vite source server on its next available port. Set
`DIAGNOSTIC_URL` to the actual URL; set `PLAYWRIGHT_MODULE` to Playwright1.65.0's
`index.mjs` and `WEBKIT_EXECUTABLE` to WebKit2388's `pw_run.sh`.

```sh
mkdir -p .local/compact-assistant-v1/memory
clang -O2 scripts/research/compact_ablation/sample.c -o .local/compact-assistant-v1/memory/sample
node scripts/research/assistant_teacher/run_browser_checks.mjs
"$EVAL_PYTHON" scripts/research/assistant_teacher/report_browser.py
LOSS_REVIEW_URL="$DIAGNOSTIC_URL/.local/compact-assistant-v1/review.html" \
  A2F_LOSS_OUTPUT=.local/compact-assistant-v1 \
  node scripts/research/assistant_teacher/verify_review.mjs
```

Browser scripts reuse the established parity/memory methods. They test the tiny
models only with the actual unchanged WebGPU runner and James animated throughout.
The larger assistant is offline only. Footprints sum WebContent+GPU processes
at20ms intervals; TTS, initial rig loading, network and browser parent are excluded.
These are desktop measurements, not physical-iPhone or full-pipeline claims.

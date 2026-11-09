# Earlier-feature residual-head ablation

[Frozen design](../../../docs/plans/compact-earlier-features.md). Generated models,
audio, caches, traces and review media stay ignored under `.local/compact-early-v1`.
Production model/defaults/runner are untouched. No deployment or publication.

This study reads the preceding data-scale corpus and exact residual-study late
feature cache. Follow their reproduction guides first. Set `A2F_RESEARCH_WORK` to
the external work root and `A2F_TEACHER_ROOT` to its
`liforma-audio2face-v3/ai-workflows/apple-silicon` folder. These dependencies are
read-only. Cache SHA checks reject modified or independently regenerated inputs.

Use `EVAL_PYTHON` with MLX0.31.2, NumPy2.5.4, ONNX1.23.2, ORT1.31.1.
`A2F_TORCH_SITE_PACKAGES` exposes Torch2.4.1 Python3.12 for export.
`TRAIN_PYTHON` uses NumPy2.2.6/Matplotlib3.10.8 for review plots.
`NODE22` is Node22.12+; synthesis/aligner assets use the established pinned
multivoice pipeline. App packages come from the existing lockfile.

```sh
"$EVAL_PYTHON" scripts/research/earlier_features/cache_features.py
"$EVAL_PYTHON" scripts/research/earlier_features/prepare_confirmation.py
"$EVAL_PYTHON" scripts/research/earlier_features/run_confirmation.py
"$EVAL_PYTHON" scripts/research/earlier_features/preflight.py
"$EVAL_PYTHON" scripts/research/earlier_features/train_heads.py --arm late
"$EVAL_PYTHON" scripts/research/earlier_features/train_heads.py --arm early
"$EVAL_PYTHON" scripts/research/earlier_features/evaluate_early.py
"$EVAL_PYTHON" scripts/research/earlier_features/build_parity.py
"$EVAL_PYTHON" scripts/research/earlier_features/check_exports.py
"$TRAIN_PYTHON" scripts/research/earlier_features/build_review.py
```

Synthesis/teacher labelling may run independently during training, but finish and
audit it before scoring. Training arms run sequentially, with no resume path;
failed trajectories must be archived and restarted. Freeze sources, input arrays,
both source scales and cache hashes before training. Development alone selects
checkpoints. Select/hash-lock both models before fresh scores.

Both heads receive135 inputs:96 source features and the identical39 normalized
v4 predictions. Late features reuse the preceding float32 cache. Early features
come immediately after the first five-tap convolution/GELU, resampled to60fps.
The original eight-block context still drives the shared base prediction. Raw
log-mel inputs and additional temporal correction are not tested here. Two
training-only source scales use the same fixed batches. No additional runtime
audio pass or model. A zero outgoing head exactly initializes both arms to v4.

The ONNX exporter materializes a deduplicated kind flag so both arms have the
same explicit graph/initializer contract. This changes serialization only. It
verifies every frozen base tensor and checks equal graph size. Native/ORT tests
check full streaming waveforms and zero inactive channels, not only cached inputs.

Run hosted-mode Vite using its next free port. Set `DIAGNOSTIC_URL` to the actual
printed URL, `PLAYWRIGHT_MODULE` to Playwright1.65.0's `index.mjs`, and
`WEBKIT_EXECUTABLE` to WebKit2388's `pw_run.sh`.

```sh
mkdir -p .local/compact-early-v1/memory
clang -O2 scripts/research/compact_ablation/sample.c -o .local/compact-early-v1/memory/sample
node scripts/research/earlier_features/run_browser_checks.mjs
"$EVAL_PYTHON" scripts/research/earlier_features/report_browser.py
LOSS_REVIEW_URL="$DIAGNOSTIC_URL/.local/compact-early-v1/review.html" \
  node scripts/research/earlier_features/verify_review.mjs
```

The research WebGPU runner preserves the first encoder activation in one extra
buffer, reuses existing convolution/frontend kernels and fuses the selected
feature source into the correction projection. Both arms use identical allocation
and dispatch topology; no extra encoder evaluation or session. Finish all GPU
training/labelling before timing/footprint trials. Animated James remains active.
Footprints sum new WebContent+GPU processes and exclude TTS, initial rig loading,
network and browser parent. They are desktop results, not physical-iPhone or
full-pipeline memory. Review audio/seek checks are not a blind perceptual study.

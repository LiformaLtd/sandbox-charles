# Shared-feature residual learning

[Frozen design](../../../docs/plans/compact-residual-learning.md).
All generated audio, models, caches and browser traces stay ignored under
`.local/compact-residual-v1`. The application and deployed v4 stay unchanged.

Use the preceding data-scale reproduction guide to obtain the pinned original
Audio2Face teacher, finetune-r4 student, James rig, source corpora, labelled gain
variants and full feature caches. Cross-repository dependencies are read-only.
Set `A2F_RESEARCH_WORK` to their work root and `A2F_TEACHER_ROOT` to its
`liforma-audio2face-v3/ai-workflows/apple-silicon` folder. Cache source hashes and
input-array hashes must match; independently modified corpora need a new study.

`EVAL_PYTHON`: MLX0.31.2, NumPy2.5.4, ONNX1.23.2, ORT1.31.1.
`A2F_TORCH_SITE_PACKAGES`: Torch2.4.1 for Python3.12 exports.
`TRAIN_PYTHON`: NumPy2.2.6/Matplotlib3.10.8 for review plots.
`NODE22`: Node22.12+; synthesis/aligner assets and app dependencies are pinned by
the existing multivoice pipeline and lockfile. No new app dependencies are added.

```sh
"$EVAL_PYTHON" scripts/research/residual_learning/cache_features.py
"$EVAL_PYTHON" scripts/research/residual_learning/prepare_confirmation.py
"$EVAL_PYTHON" scripts/research/residual_learning/run_confirmation.py
"$EVAL_PYTHON" scripts/research/residual_learning/preflight.py
"$EVAL_PYTHON" scripts/research/residual_learning/inspect_residuals.py
"$EVAL_PYTHON" scripts/research/residual_learning/train_heads.py --arm control
"$EVAL_PYTHON" scripts/research/residual_learning/train_heads.py --arm residual
"$EVAL_PYTHON" scripts/research/residual_learning/evaluate_residual.py
"$EVAL_PYTHON" scripts/research/residual_learning/explained_error.py
"$EVAL_PYTHON" scripts/research/residual_learning/build_parity.py
"$EVAL_PYTHON" scripts/research/residual_learning/check_exports.py
"$TRAIN_PYTHON" scripts/research/residual_learning/build_review.py
```

Synthesis and teacher labelling may finish independently during training, but
selection locks must precede any fresh scores. Both arms train sequentially.
All base layers remain fixed. Caches store the full60-frame interpolated learned
features, float32, with central30-frame valid crops for supervision. There is no
extra temporal convolution, state, input context or audio encoder. Feature scales
use32 fixed training batches. The direct GELU-pair head and zero residual head
start with the same function/loss and equal16,839 trainable parameters. Equal
functions do not imply identical internal gradients or equally expressive
architectures: residual skip supplies the known base prediction separately.

Training has no resume path: a failed trajectory must be archived and restarted.
Source/data locks, development histories and selected head checkpoints are kept.
The original coefficient/geometry/contact objective is applied to the corrected
prediction, mathematically equivalent to learning coefficient residuals for a
coefficient squared loss, while preserving the existing nonlinear geometry and
closure terms. No extra residual-MSE objective replaces the established loss.

For an initial browser parity check, `build_parity.py --initial` writes fixtures
under `browser/preflight`; point the generic verifier at the two preflight ONNX
files and `residual-webgpu.js` through `A2F_PARITY_MODELS`/`A2F_PARITY_RUNNERS`.
These checks are numerical controls, not performance measurements during training.

Run hosted-mode Vite on its next available port and set `DIAGNOSTIC_URL` to its
actual URL. Set `PLAYWRIGHT_MODULE` to Playwright1.65.0's `index.mjs` and
`WEBKIT_EXECUTABLE` to WebKit2388's `pw_run.sh`.

```sh
mkdir -p .local/compact-residual-v1/memory
clang -O2 scripts/research/compact_ablation/sample.c -o .local/compact-residual-v1/memory/sample
node scripts/research/residual_learning/run_browser_checks.mjs
"$EVAL_PYTHON" scripts/research/residual_learning/report_browser.py
LOSS_REVIEW_URL="$DIAGNOSTIC_URL/.local/compact-residual-v1/review.html" \
  node scripts/research/residual_learning/verify_review.mjs
```

Finish all GPU training and teacher labelling before isolated browser benchmarks.
The research runner reuses the existing audio and convolution kernels and fuses
the old projection, shared-feature head and final correction into one workgroup
per animation frame. No second model/session or extra readback is used. Production
runner files are not modified. The final head models are2.321912MB, about3.11%
larger than v4. Browser verification uses actual animated James and prerecorded
STA; process footprints exclude TTS, initial rig loading and browser parent, and
are not physical-iPhone or full-pipeline measurements. No publication/deployment.

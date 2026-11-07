# Teacher-feature distillation on expanded speech

Read the [frozen experiment](../../../docs/plans/compact-feature-distillation.md).
This is offline research. The current browser architecture and published v4
scale-r3 stay unchanged. It revisits an inconclusive earlier small-corpus trial
using normalized teacher features and the much larger current training corpus.

Run from the application root with the prior data-scale artifacts available at
`.local/compact-scale-v1` and the earlier coverage/multivoice/gain/rig/packed
artifacts installed as described in their research READMEs. The current v4 is
`compact-scale-v1/expanded-long/model.onnx`, SHA recorded in the frozen plan.
Dependencies match the data-scale experiment: PyTorch2.4.1/NumPy2.2.6 training,
MLX0.31.2, ONNX1.23.2/ORT1.31.1 evaluation, Node22.22.3 and desktop WebKit2388.
Review plots additionally use Matplotlib3.10.8 in the training interpreter.
The source exports the same waveform model; MLX is not a browser dependency.

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/research-work
export A2F_FEATURE_OUTPUT="$PWD/.local/compact-feature-v1"
export A2F_TEACHER_ROOT=/absolute/path/to/ai-workflows/apple-silicon
export TRAIN_PYTHON=/absolute/path/to/torch-python
export EVAL_PYTHON=/absolute/path/to/onnx-python
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx-python/site-packages
export NODE22=/absolute/path/to/node22
```

The optional ONNX site-packages path is a compatibility aid for the original
split environments. Use compatible versions and rerun checks in a new setup.
TF32 is disabled before importing MLX. `A2F_SCALE_OUTPUT` must remain unset or
point to the existing data-scale input root: the imported read-only Dataset
loader reads those original arrays, not the new experiment's output directory.

`prepare_confirmation.py` records every retained prior manifest and excludes
all its normalized texts. It contains the original machine's historical-artifact
root explicitly; migrate that inventory before a new run, freeze the new source
and do not claim identical hashes. Historical manifests contain absolute audio
paths too. The 12 new broad prompts use the prior pinned CMU ARCTIC text source;
its CMU license/notice and source commit remain in the data-scale provenance.
Only newly synthesized Phonon/Supertonic speech is used.

```sh
"$EVAL_PYTHON" scripts/research/feature_distillation/prepare_confirmation.py
"$EVAL_PYTHON" scripts/research/feature_distillation/run_confirmation.py
"$EVAL_PYTHON" scripts/research/feature_distillation/cache_hints.py
"$TRAIN_PYTHON" scripts/research/feature_distillation/check.py
"$EVAL_PYTHON" scripts/research/feature_distillation/freeze_training.py
"$TRAIN_PYTHON" scripts/research/feature_distillation/train.py --arm control
"$TRAIN_PYTHON" scripts/research/feature_distillation/train.py --arm distill
"$EVAL_PYTHON" scripts/research/feature_distillation/build_review.py --freeze-only
"$EVAL_PYTHON" scripts/research/feature_distillation/evaluate_experiment.py
"$TRAIN_PYTHON" scripts/research/feature_distillation/build_review.py
"$EVAL_PYTHON" scripts/research/feature_distillation/build_parity.py
```

Confirmation synthesis can overlap hint caching/training, but lock visual cases
and evaluator before fresh scoring. Never use confirmation to change weights,
loss coefficient or checkpoint selection. The evaluator locks both exports and
checks identical sampler hashes before reading fresh scores. Every gain variant
gets its own teacher output. Cached hints use the existing last encoder layer
at 60fps, aligned with the same original/augmented waveform and emitted crop.
The 28,845 primary hint windows occupy about 1.33GB as offline f16 maps; they are
not shipped to the browser. The affine hint head is removed at export.

Both arms instantiate the same random auxiliary head and share seed 37,
3,720 updates and development-only selection. Distillation alone ramps the
normalized cosine loss to 0.1 over 372 updates. Historical replay retains ordinary
animation losses with no hint supervision. Completion records include exact
sampler hashes. Resume restores optimizer/model/RNG and reconstructs the complete
sampler hash, including the completed prefix. A completed run refuses overwriting.

## Browser checks

Start hosted-mode source Vite using its next free port; set the printed URL.
Run after all teacher and training jobs finish. This checks the normal animated
James page plus standalone current/control/distilled STA inference; TTS is not
loaded for memory measurements.

```sh
export DIAGNOSTIC_URL=http://127.0.0.1:YOUR_PORT
export PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs
export WEBKIT_EXECUTABLE=/absolute/path/to/webkit/pw_run.sh
export A2F_PARITY_MODELS='{"v4":"/.local/compact-scale-v1/expanded-long/model.onnx","control":"/.local/compact-feature-v1/control/model.onnx","distill":"/.local/compact-feature-v1/distill/model.onnx"}'
"$NODE22" scripts/research/feature_distillation/verify_browser.mjs
export SCALE_REVIEW_URL="$DIAGNOSTIC_URL/.local/compact-feature-v1/review.html"
"$NODE22" scripts/research/feature_distillation/verify_review.mjs
cc -O2 scripts/research/compact_ablation/sample.c -o "$A2F_FEATURE_OUTPUT/sample-footprint"
export A2F_MEMORY_SAMPLER="$A2F_FEATURE_OUTPUT/sample-footprint"
"$NODE22" scripts/research/feature_distillation/measure_memory.mjs
```

Preserve three fresh settled browser runs separately; warm five calls and wait
for rig startup before timing. Compare complete native-ORT/browser tracks,
invalid/all-zero frames and byte-identical first-clip repeats. Explicit model
buffer counts exclude browser/driver/renderer overhead. The macOS physical
footprint sampler sums new WebContent/GPU processes concurrently at 20 ms; it can
miss brief peaks and excludes TTS. An emulated iPhone UA is not a physical iPhone.

```sh
"$EVAL_PYTHON" -m unittest discover -s scripts/research/feature_distillation -p 'test_*.py'
"$EVAL_PYTHON" -m compileall -q scripts/research/feature_distillation
"$NODE22" --check scripts/research/feature_distillation/verify_browser.mjs
"$NODE22" --check scripts/research/feature_distillation/verify_review.mjs
"$NODE22" --check scripts/research/feature_distillation/measure_memory.mjs
git diff --check
```

Keep models, audio, feature maps, frozen source/recipe snapshots, teacher notices,
optimizer/RNG checkpoints and raw reports under the ignored output root. Do not
commit weights, generated speech or dependencies. No CDN/Vercel release here.

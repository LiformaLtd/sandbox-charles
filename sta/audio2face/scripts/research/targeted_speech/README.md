# Targeted difficult speech, with matched broad-data and compute controls

Read the [frozen plan](../../../docs/plans/compact-targeted-speech.md) and
[results](../../../docs/plans/compact-targeted-speech-results.md).
This experiment starts from released v4 finetune-r4. It does not alter the
browser graph, ship phoneme timings, use teacher-feature supervision or deploy
an experimental model. All synthesis, alignment, mining and teacher work is
offline. James continues animating in the browser benchmarks.

## Inputs and environment

Reproduce the prior data-scale and feature-distillation artifact setup first.
Required ignored inputs: `.local/compact-scale-v1`, `.local/compact-feature-v1`,
coverage/multivoice/gain/packed-diagnostic assets, full teacher, TinyAligner,
James rig/channels and the historical v3 replay roots in the research workspace.
The pinned source assets, synthesis settings, fitter and teacher are unchanged.

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/research-work
export A2F_TARGET_OUTPUT="$PWD/.local/compact-targeted-v1"
export A2F_TEACHER_ROOT=/absolute/path/to/ai-workflows/apple-silicon
export TRAIN_PYTHON=/absolute/path/to/torch-python
export EVAL_PYTHON=/absolute/path/to/onnx-python
export NODE22=/absolute/path/to/node22
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx-python/site-packages
```

Recorded stack: MLX0.31.2 float32, TF32 disabled before import; PyTorch2.4.1 /
NumPy2.2.6 for export; ONNX1.23.2 / ORT1.31.1 / NumPy2.5.4 for evaluation;
Matplotlib3.10.8 for review plots; Node22.22.3 and desktop WebKit2388. Keep a
compatible environment and rerun checks on other versions. The optional
site-packages setting exposes ONNX/MLX to the original split training environment.
`common.py` sets `A2F_SCALE_OUTPUT` to this experiment's output before importing
the unchanged memory-map loader; its historical input directories are read-only
symlinks to data-scale arrays. Do not accidentally point output at prior studies.

Prior manifests contain absolute waveform paths. Migrate and retain the complete
prior transcript inventory for a new replication rather than omitting missing
files. New corpus preparation refuses missing prior inventory. The CMU ARCTIC
text, source hash/commit and license are retained with the prior data-scale
artifacts. No original human recordings are reused. New text is unchanged CMU
ARCTIC, not an automatically generated assertion about natural language quality.

## Freeze corpus and run

```sh
"$EVAL_PYTHON" scripts/research/targeted_speech/prepare_corpus.py
"$EVAL_PYTHON" scripts/research/targeted_speech/audit_data.py
"$EVAL_PYTHON" scripts/research/targeted_speech/run_corpus.py
"$EVAL_PYTHON" scripts/research/targeted_speech/assemble.py
"$TRAIN_PYTHON" scripts/research/targeted_speech/build_features.py
"$EVAL_PYTHON" scripts/research/targeted_speech/freeze_training.py
"$TRAIN_PYTHON" scripts/research/targeted_speech/train.py --arm control
"$TRAIN_PYTHON" scripts/research/targeted_speech/train.py --arm broad
"$TRAIN_PYTHON" scripts/research/targeted_speech/train.py --arm targeted
```

After `audit_data.py` passes, `postprocess.py` may run alongside synthesis: it
waits for completion, assembles, caches, freezes and runs the three arms, stopping
if a shard fails. It never scores fresh confirmation automatically.

Mining uses current-v4 TRAINING residuals only and a deterministic support/
shrinkage rule. Teacher errors are measured at actual aligned phone intervals;
text candidates use stress-free dictionary phone triples across word boundaries.
This is a proxy for difficult acoustics, not proof that a text context uniquely
causes the residual. Corpus audit records chosen context scores, support,
reserved evaluation pool, broad/targeted enrichment and sentence lengths.

Broad texts are hash-selected first. Targeted texts are selected from the
remaining training pool, matching word count within one when possible. Both
arms use paired voice/gain assignments. The final 72-text schedule has exactly
36 recordings per voice and 12 per voice/gain, for 288 independent recordings per
addition. A pre-synthesis preparation draft caught an imbalance in an eight-step
voice cycle; it was archived locally and corrected to the final balanced
schedule before any audio was synthesized. The archived draft is not an input.

Eight bounded synthesis/alignment shard jobs share at most two teacher slots.
Each transformed waveform is independently labelled. Labels retain the teacher's
unused speech features for provenance but this experiment trains only the
existing animation objective. Dataset assembly verifies waveform hashes,
complete target tracks and identical teacher recipes across shards.

Original-data control keeps 48 historical-primary windows per update. Broad and
targeted retain the identical 32 historical-primary windows and replace 16 others
with voice-balanced novel examples. Replay is the same 16 clean windows. Separate
old/replay/new random streams avoid changing replay when novel pool lengths
differ. Only control uses the final 16 old draws; shared draw hashes are audited
before scoring. New-data windows are intentionally oversampled, not uniformly
mixed with the old corpus. All arms keep seed 37, 3,720 updates and development-only
selection. Complete checkpoint/optimizer/RNG states support resume.

## Lock scoring and browser comparisons

```sh
"$EVAL_PYTHON" scripts/research/targeted_speech/build_review.py --freeze-only
"$EVAL_PYTHON" scripts/research/targeted_speech/evaluate_experiment.py
"$TRAIN_PYTHON" scripts/research/targeted_speech/build_review.py
"$EVAL_PYTHON" scripts/research/targeted_speech/build_parity.py
```

Freeze visual cases and evaluator before the first fresh score. All three
checkpoints are locked first. Report both combined and broad/enriched strata,
engine/gain/voice/closure/opening/timing slices, and inspected historical/feature
regression sets. Keep failed screens intact; no confirmation-driven prompt,
coefficient, seed, threshold or checkpoint adjustment.

Prepare or link the `.local/rig.bin` and `.local/rig.json` files as well as the
model directories; check `/assets/rig.json` returns 200. Start the app's hosted-mode
source Vite server and use its printed free port.
Run benchmarks after synthesis, training and native reference generation end:

```sh
export DIAGNOSTIC_URL=http://127.0.0.1:YOUR_PORT
export PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs
export WEBKIT_EXECUTABLE=/absolute/path/to/webkit/pw_run.sh
export A2F_PARITY_MODELS='{"v4":"/.local/compact-feature-v1/control/model.onnx","control":"/.local/compact-targeted-v1/control/model.onnx","broad":"/.local/compact-targeted-v1/broad/model.onnx","targeted":"/.local/compact-targeted-v1/targeted/model.onnx"}'
"$NODE22" scripts/research/targeted_speech/verify_browser.mjs
export SCALE_REVIEW_URL="$DIAGNOSTIC_URL/.local/compact-targeted-v1/review.html"
"$NODE22" scripts/research/targeted_speech/verify_review.mjs
cc -O2 scripts/research/compact_ablation/sample.c -o "$A2F_TARGET_OUTPUT/sample-footprint"
export A2F_MEMORY_SAMPLER="$A2F_TARGET_OUTPUT/sample-footprint"
"$NODE22" scripts/research/targeted_speech/measure_memory.mjs
```

Preserve three fresh settled timing/parity runs, waiting for the rig and five
warm-up calls. Explicit GPU bytes do not include driver allocations. The macOS
physical-footprint sampler sums new WebContent/GPU processes at 20 ms during
short/long/repeated prerecorded STA with normal idle animation. TTS, browser and
network processes are excluded; brief peaks can be missed. Neither desktop
process measurements nor an emulated iPhone UA guarantee physical-iPhone memory.

```sh
"$EVAL_PYTHON" -m compileall -q scripts/research/targeted_speech
"$NODE22" --check scripts/research/targeted_speech/verify_browser.mjs
"$NODE22" --check scripts/research/targeted_speech/verify_review.mjs
"$NODE22" --check scripts/research/targeted_speech/measure_memory.mjs
git diff --check
```

Keep generated text manifests, source/license/protocol snapshots, waveforms,
teacher labels, maps, model/optimizer/RNG checkpoints and raw evaluation/browser
reports under the ignored output root. Git contains research code/docs only.
No model CDN upload or Vercel change follows automatically from this experiment.

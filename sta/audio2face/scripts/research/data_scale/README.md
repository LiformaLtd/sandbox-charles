# Compact learnability and data scale

Read the [results](../../../docs/plans/compact-data-scale-results.md),
[frozen diagnostic](../../../docs/plans/compact-data-scale.md),
[matched grid](../../../docs/plans/compact-data-scale-grid.md) and
[post-pilot extension](../../../docs/plans/compact-scale-seed-stability.md).
Run from this application's root. This is an offline research pipeline, not an
extra production dependency. Generated audio, weights and caches stay ignored.

## Prerequisites and provenance

Reproduce the earlier coverage-r2 study first: its pinned assets, voices, teacher
and training model are inputs here. See `../coverage/` and the earlier learning
notes; this directory is not a standalone replacement for that setup. Required
local artifacts include `.local/compact-coverage-v2`, the prior multivoice,
confirmation, gain/capacity/state/temporal manifests, and `arkit-student`,
`arkit-student-v2`, `arkit-student-v3` (including extra) in the research workspace.
The rig metrics, channels, full teacher and TinyAligner are offline dependencies.
TinyAligner is not added to browser inference. Preserve the historical transcript
inventory; omitting it changes the corpus exclusion policy.

Set these to real absolute paths on your machine:

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/research-work
export A2F_SCALE_OUTPUT="$PWD/.local/compact-scale-v1"
export A2F_TEACHER_ROOT=/absolute/path/to/ai-workflows/apple-silicon
export TRAIN_PYTHON=/absolute/path/to/torch-python
export EVAL_PYTHON=/absolute/path/to/onnx-python
export NODE22=/absolute/path/to/node22
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx-python/site-packages
```

Recorded versions: training PyTorch 2.4.1 / NumPy 2.2.6, MLX 0.31.2;
evaluation NumPy 2.5.4 / ONNX 1.23.2 / ORT 1.31.1; Node 22.22.3.
The optional site-packages path exposes ONNX/MLX to the training interpreter
without replacing the already-imported training NumPy. Prefer a properly
compatible unified environment for a new replication and repeat backend checks.
The training entry points disable TF32 before importing MLX. Do not use the
failed Torch MPS route from the diagnostic logs.

## Memorization input preparation

The original four-clip preparation was a manifest/label copy, not a separate
trainer. Select the following *training*, original-level stems from
`.local/compact-coverage-v2/variants/manifest.json` in this order:

1. `coverage-train-phonon-easton-000-original`
2. `coverage-train-phonon-zoey-018-original`
3. `coverage-train-supertonic-m1-034-original`
4. `coverage-train-supertonic-f1-064-original`

Write unchanged rows to `$A2F_SCALE_OUTPUT/memorization/manifest.json`, retain
absolute audio paths, copy matching `variants/labels/*.npz` to `memorization/labels`
and copy `coverage/sources/model.py` to `memorization/sources/model.py`.
Create `memorization/protocol.json` with `baseline` (coverage-r2 ONNX absolute
path), `baseline_sha256`, `manifest_sha256`, `trainer_sha256` (memorize.py),
`plan_sha256` (diagnostic plan), `model_source_sha256`, and `audio`/`labels`
dictionaries mapping each stem to its waveform/NPZ SHA-256. JSON hashes are byte
hashes, not hashes of normalized objects. Freeze these before optimization.
Baseline SHA is recorded in the results. Never edit the selected examples after
seeing a diagnostic result.

```sh
"$TRAIN_PYTHON" scripts/research/data_scale/memorize.py --objective mixed
"$TRAIN_PYTHON" scripts/research/data_scale/memorize.py --objective fit-mse
```

Both completed diagnostic results must pass before corpus preparation proceeds.
Reruns require a separate output root; completed runs refuse overwriting.

## Corpus, teacher and matched training

Put the unchanged `cmuarctic.data` from the pinned source URL in
`$A2F_SCALE_OUTPUT/text-sources/`, retaining the CMU license notice alongside it.
Pinned mirror commit: `c51cb971036a127359b43c621f26ca5081eab13e` in
`rhasspy/dataset-voice-kathleen`. The selected text source and recipe are frozen
by `prepare_corpus.py`. Source licensing is documented in the results.

`audit_disjoint.py` contains the original machine's prior-artifact root explicitly.
For a new replication, adapt it to your retained historical manifests *before*
freezing any protocol; archive the changed source and do not claim identical
source hashes. Other manifests also contain absolute audio paths: migrate them
and hash the migrated inputs rather than silently treating them as identical.

```sh
"$EVAL_PYTHON" scripts/research/data_scale/prepare_corpus.py
"$EVAL_PYTHON" scripts/research/data_scale/audit_disjoint.py
"$EVAL_PYTHON" scripts/research/data_scale/run_corpus.py
"$EVAL_PYTHON" scripts/research/data_scale/assemble.py
"$TRAIN_PYTHON" scripts/research/data_scale/features.py
"$TRAIN_PYTHON" scripts/research/data_scale/check_backend.py
"$EVAL_PYTHON" scripts/research/data_scale/freeze_training.py
"$TRAIN_PYTHON" scripts/research/data_scale/train.py --data original --seed 37
"$TRAIN_PYTHON" scripts/research/data_scale/train.py --data expanded --seed 37
"$EVAL_PYTHON" scripts/research/data_scale/lock_selection.py
```

Corpus synthesis is resumable (12 bounded shard workers, max two teacher jobs).
Inspect progress/logs; never remove a pipeline lock while its owner is running.
Every gain-altered waveform gets independently inferred/fitted teacher labels.
`postprocess.py` can wait for synthesis and execute assembly/cache/freeze/training,
but first run and pass the numerical backend preflight manually as above.

Before fresh scoring, archive `evaluate.py` as `sources/evaluate-scale.py` and
freeze the planned visual cases using `build_review.py --freeze-only`. Then score once:

```sh
"$EVAL_PYTHON" scripts/research/data_scale/build_review.py --freeze-only
"$EVAL_PYTHON" scripts/research/data_scale/evaluate.py
"$EVAL_PYTHON" scripts/research/data_scale/build_review.py
"$EVAL_PYTHON" scripts/research/data_scale/build_parity.py
```

Evaluate writes prediction caches and all four arms plus v4; `summary.json` keeps
the original pilot screen. Do not change the threshold or overwrite failure with
an average from later experiments. The review case selection is frozen before
scoring; its model tracks can be regenerated after export without changing cases.

For the separately documented seed extension, prepare independent roots
`repeats/s73` and `repeats/s101`: retain byte-identical training protocol, share
read-only datasets/features and the frozen model source, but give each root its
own trajectories, selections and predictions. Invoke `train.py` with `--seed 73`
or `--seed 101`, both data arms, then lock/score each root by setting
`A2F_SCALE_OUTPUT` accordingly. Reset output to the main root and run
`summarize_seeds.py`. All three seeds must remain in the report; never choose the
lowest-confirmation-error seed for publication.

## Browser, review and memory

Start the normal hosted-mode source Vite server using the app README. Use its
printed free port; there is no research-specific fixed port. Set:

```sh
export DIAGNOSTIC_URL=http://127.0.0.1:YOUR_PORT
export PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs
export WEBKIT_EXECUTABLE=/absolute/path/to/webkit/pw_run.sh
export A2F_PARITY_MODELS='{"v4":"/.local/compact-coverage-v2/coverage/runs/control-replay-s37/model.onnx","original-short":"/.local/compact-scale-v1/original-short/model.onnx","original-long":"/.local/compact-scale-v1/original-long/model.onnx","expanded-short":"/.local/compact-scale-v1/expanded-short/model.onnx","expanded-long":"/.local/compact-scale-v1/expanded-long/model.onnx"}'
"$NODE22" scripts/research/data_scale/verify_browser.mjs
export SCALE_REVIEW_URL="$DIAGNOSTIC_URL/.local/compact-scale-v1/review.html"
"$NODE22" scripts/research/data_scale/verify_review.mjs
cc -O2 scripts/research/compact_ablation/sample.c -o "$A2F_SCALE_OUTPUT/sample-footprint"
export A2F_MEMORY_SAMPLER="$A2F_SCALE_OUTPUT/sample-footprint"
"$NODE22" scripts/research/data_scale/measure_memory.mjs
```

Run three fresh browser instances after teacher/training activity ends, preserving
each parity report/output separately. The verifier waits for the rig and warms
five calls. Compare full tracks, finite values, repeated first-clip bytes and
explicit buffer allocations, not a single window. Optional
`A2F_ACTIVATION_OUTPUT`/`A2F_ACTIVATION_ASSET_PATH` select another fixture folder.

The memory sampler is macOS-specific and sums newly launched WebContent/GPU
processes at 20 ms intervals. It runs prerecorded inference with normal idle
animation, **without TTS**. It cannot prove full-pipeline iPhone safety. The review
page holds head pose still for mouth comparison; production idle animation stays.

## Verification and artifact handling

```sh
"$EVAL_PYTHON" -m unittest discover -s scripts/research/data_scale -p 'test_*.py'
"$EVAL_PYTHON" -m compileall -q scripts/research/data_scale
"$NODE22" --check scripts/research/data_scale/verify_browser.mjs
"$NODE22" --check scripts/research/data_scale/verify_review.mjs
"$NODE22" --check scripts/research/data_scale/measure_memory.mjs
git diff --check
```

Audit every source and array hash in `training-protocol.json` after the study;
retain the protocol, corpus source/license, manifests, source snapshots, complete
checkpoint/optimizer/RNG states, selection locks and reports with the ignored
local artifacts. No weights or audio belong in Git. This phase does not upload
models, change production v4 or deploy Vercel.

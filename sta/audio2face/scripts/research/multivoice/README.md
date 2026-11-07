# Multi-voice compact research

See [the study definition](../../../docs/plans/compact-multivoice.md). These are
offline scripts; production browser modules do not import them. Generated audio,
alignments, labels, feature caches, checkpoints and exports stay outside Git.

Use Node 22.22.3, Phonon 0.6.0, ONNX Runtime Web 1.31.1; training Python 3.12,
PyTorch 2.4.1 and NumPy 2.2.6; teacher/evaluation Python 3.12 with MLX, ONNX 1.23.2,
ONNX Runtime 1.31.1 and NumPy 2.5.4, matching the previous study. Different pinned
environments are supported through explicit executable paths. Do not install or
upgrade dependencies in another project's checkout.

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/earlier-research-work
export A2F_TEACHER_ROOT=/absolute/path/to/compatible/apple-silicon
export A2F_DICT_ROOT=/absolute/path/to/audio2face-dict
export A2F_ALIGNER_ROOT=/absolute/path/to/tinyaligner-fast
# Optional: defaults to this app's ignored .local/compact-multivoice directory.
export A2F_MULTIVOICE_OUTPUT=/absolute/path/to/new-study-directory
export TRAIN_PYTHON=/absolute/path/to/training/python
export EVAL_PYTHON=/absolute/path/to/teacher-and-evaluation/python
# Only if ONNX is in a separate compatible environment:
export A2F_ONNX_SITE_PACKAGES=/absolute/path/to/onnx/site-packages

"$TRAIN_PYTHON" scripts/research/multivoice/prepare.py
node scripts/research/multivoice/synthesize.mjs phonon
node scripts/research/multivoice/synthesize.mjs supertonic
"$EVAL_PYTHON" scripts/research/multivoice/verify.py
node --experimental-strip-types scripts/research/multivoice/align.mjs
"$EVAL_PYTHON" scripts/research/multivoice/label.py
"$TRAIN_PYTHON" scripts/research/multivoice/test_contract.py
"$TRAIN_PYTHON" scripts/research/multivoice/train.py --arm control
"$TRAIN_PYTHON" scripts/research/multivoice/train.py --arm phoneme
"$TRAIN_PYTHON" scripts/research/multivoice/train.py --arm features
"$EVAL_PYTHON" scripts/research/multivoice/evaluate.py v3 \
  "$A2F_RESEARCH_WORK/arkit-student-v3/james-arkit-student-v3.onnx"
"$EVAL_PYTHON" scripts/research/multivoice/evaluate.py control \
  "$A2F_MULTIVOICE_OUTPUT/runs/control-s37/model.onnx"
# Repeat evaluation for phoneme and features; --split dev reports validation.
```

Run commands from the app directory, using the selected Node executable in PATH.
The two synthesis engines may run in separate processes. Wait for both to finish
before alignment. Do not launch the same synthesis engine twice in one output
directory. An interrupted process may leave a lock; verify its recorded PID is
dead before removing that lock. Synthesis, alignment and labeling resume checked
per-clip artifacts. Training saves model, optimizer, schedule, minibatch RNG and
best checkpoint every epoch, and can resume an identical recipe. A completed
export is not silently overwritten.

`prepare.py` expects the app's existing `.local/phonon` and `.local/supertonic`
assets, the earlier compact manifests, the dictionary project's lexicon/alignment
graph and label fitter, and TinyAligner Fast source. It downloads only the small
additional preset embeddings/styles from pinned revisions. Frozen source files
use explicit `.ts` imports for Node's TypeScript stripping. Full model downloads
and compatible native teacher setup are described in the earlier learnings.

`test_contract.py` verifies that auxiliary losses train the existing encoder,
the training wrapper preserves its prediction, ONNX drops the auxiliary heads,
and partial streaming crops invert the teacher's sample grid correctly. The
teacher instrumentation parity check must pass before label generation.

Inference checkpoints contain only the unchanged student. Evaluation caches are
keyed by model and audio hashes. Test evaluation is descriptive and must not be
used to select checkpoints, hyperparameters or repeatedly tuned combinations.

## Follow-up diagnostics and closure timing

```sh
"$TRAIN_PYTHON" scripts/research/multivoice/easton_control.py
"$EVAL_PYTHON" scripts/research/multivoice/evaluate.py easton-control \
  "$A2F_MULTIVOICE_OUTPUT/runs/easton-control-s37/model.onnx"
"$TRAIN_PYTHON" scripts/research/multivoice/train.py --arm control --clean-replay
"$EVAL_PYTHON" scripts/research/multivoice/evaluate.py replay \
  "$A2F_MULTIVOICE_OUTPUT/runs/control-replay-s37/model.onnx"
"$EVAL_PYTHON" scripts/research/multivoice/gain_diagnostic.py
"$TRAIN_PYTHON" scripts/research/multivoice/test_closure_metrics.py
"$TRAIN_PYTHON" scripts/research/multivoice/summarize.py
```

The Easton-only and clean-replay runs were post-pilot engineering diagnostics.
They are not new confirmation sets. Replay uses only consistently fitted older
v3/v3-extra labels and an Easton development-set term, not old v1/v2 fits.
`sources/train-first-pilot.py` preserves the completed pilot recipe; the current
trainer adds replay support and therefore has a different source fingerprint.
Resume an archived run with its frozen source or reproduce in a fresh output
directory. `easton_control.py` generates and archives an explicit modified copy;
it does not edit the original trainer or another checkout.

The historical Easton comparison reuses `compact_ablation/evaluate_confirmation.py`:
point `A2F_ABLATION_OUTPUT` at the study output, provide its `confirmation/`
manifest/labels and a root-level `<name>.onnx` copy, then run the evaluator with
that name. The artifact JSON retains pooled historical metrics separately from
the new test set's macro voice metrics.

Current `evaluate.py` keeps the old event counts under their explicit legacy
policy and adds `closure_timing_audit`. Primary association uses 100 ms
pre-padding and 35 ms post-padding, with ambiguous/shared/truncated contacts
reported separately. The predicted episode must match the teacher's actual
contact, not simply fall anywhere near a phoneme. Extra 45/150 ms association
audits, signed timing errors and a stricter IoU/onset score are included. Training
targets are unshifted teacher geometry, not phone-centred contact masks.

## Browser and synchronized rendering

```sh
"$TRAIN_PYTHON" scripts/research/multivoice/prepare_browser.py
export A2F_ABLATION_OUTPUT="$A2F_MULTIVOICE_OUTPUT/browser"
"$EVAL_PYTHON" scripts/research/compact_ablation/fixtures.py baseline control data folded replay
node node_modules/vite/bin/vite.js --config "$A2F_ABLATION_OUTPUT/vite.config.mjs" --mode hosted
# In another terminal: use the actual next-free Vite URL and your Playwright module.
export BENCH_URL=http://127.0.0.1:ACTUAL_PORT
export PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs
export WEBKIT_EXECUTABLE=/absolute/path/to/webkit/pw_run.sh
BENCH_REPEATS=2 node "$A2F_ABLATION_OUTPUT/browser_bench.mjs" baseline data replay
BENCH_TTS=supertonic BENCH_REPEATS=1 node "$A2F_ABLATION_OUTPUT/browser_bench.mjs" replay
"$TRAIN_PYTHON" scripts/research/multivoice/build_review.py
```

The reference harness aliases are recorded: `control` = data-only, `data` =
phoneme supervision, `folded` = feature distillation, `replay` = clean replay.
These names are research harness routing only. They do not rename production
models. `prepare_browser.py` requires the completed four candidate exports.
Local pinned Supertonic assets are copied because Vite does not run Vercel's
redirects. Run timing/memory tests after all training stops. Inspect `/review.html`
for six predetermined speech clips and six synchronized James lanes. The normal
demo's model selection, idle animation and deployed assets are untouched.

Full interpretation, exclusions and measured results are in
[the results report](../../../docs/plans/compact-multivoice-results.md).

## V4 candidate: independent seeds and fresh confirmation

The frozen [validation plan](../../../docs/plans/compact-v4-validation.md) and
[results](../../../docs/plans/compact-v4-validation-results.md) document the next
stage. Keep the pilot output directory for training and use a separate output
for fresh confirmation. Do not run timing/memory checks alongside training.

```sh
"$TRAIN_PYTHON" scripts/research/multivoice/train.py --arm control --clean-replay --seed 73
"$TRAIN_PYTHON" scripts/research/multivoice/train.py --arm control --clean-replay --seed 101
export A2F_CONFIRMATION_ROOT="$PWD/.local/compact-v4-confirmation"
export A2F_VALIDATION_OUTPUT="$PWD/.local/compact-v4-validation"
A2F_MULTIVOICE_OUTPUT="$A2F_CONFIRMATION_ROOT" \
  A2F_MULTIVOICE_VOICES=scripts/research/multivoice/confirmation-voices.json \
  A2F_MULTIVOICE_PROMPTS=scripts/research/multivoice/confirmation-prompts.json \
  "$TRAIN_PYTHON" scripts/research/multivoice/prepare.py
"$TRAIN_PYTHON" scripts/research/multivoice/freeze_confirmation.py
A2F_MULTIVOICE_OUTPUT="$A2F_CONFIRMATION_ROOT" node scripts/research/multivoice/synthesize.mjs phonon
A2F_MULTIVOICE_OUTPUT="$A2F_CONFIRMATION_ROOT" node scripts/research/multivoice/synthesize.mjs supertonic
A2F_MULTIVOICE_OUTPUT="$A2F_CONFIRMATION_ROOT" node --experimental-strip-types scripts/research/multivoice/align.mjs
A2F_MULTIVOICE_OUTPUT="$A2F_CONFIRMATION_ROOT" "$EVAL_PYTHON" scripts/research/multivoice/verify.py
A2F_MULTIVOICE_OUTPUT="$A2F_CONFIRMATION_ROOT" "$EVAL_PYTHON" scripts/research/multivoice/label.py
"$EVAL_PYTHON" scripts/research/multivoice/validate_candidate.py --stage select
"$EVAL_PYTHON" scripts/research/multivoice/validate_candidate.py --stage evaluate
"$TRAIN_PYTHON" scripts/research/multivoice/prepare_validation_browser.py
A2F_ABLATION_OUTPUT="$A2F_VALIDATION_OUTPUT/browser" \
  "$EVAL_PYTHON" scripts/research/compact_ablation/fixtures.py baseline candidate
node node_modules/vite/bin/vite.js --config "$A2F_VALIDATION_OUTPUT/browser/vite.config.mjs" --mode hosted
# In another terminal, set BENCH_URL to its printed URL; preserve the same env.
node scripts/research/multivoice/repeated_browser.mjs
"$TRAIN_PYTHON" scripts/research/multivoice/build_confirmation_review.py
node scripts/research/multivoice/verify_confirmation_review.mjs
```

Selection verifies identical recipes/input hashes across seeds and freezes the
development-selected representative before test evaluation. The evaluator
compares all three seeds with v3 on the fresh set, the old pilot and historical
Easton corpus. It saves a crossed seed/voice/text bootstrap and reports every
predeclared engineering check. It does not promote the model automatically.

`A2F_MULTIVOICE_PILOT` can relocate the pilot artifacts. Historical Easton
evaluation requires the pilot's `confirmation/` manifest/labels, as described
above. Confirmation definitions use four new voices and 24 new texts, without
changing the training corpus. `freeze_confirmation.py` rejects a changed
protocol and preserves existing protocol bytes. The representative is copied to
`$A2F_VALIDATION_OUTPUT/v4-candidate.onnx`, with its hash in `selection.json`.

The repeated browser test runs three different generations per model/engine on
one prepared page, checks fresh non-silent audio/inference, stable model GPU
allocations, navigation count and idle motion, and samples process footprints.
It covers desktop WebKit, not a physical iPhone or a long-duration soak. The
review verifier waits until the actual rendered frame matches each seek target
before capturing screenshots; audio seeking is asynchronous.

## Publishing and checking the demo candidate

`publish_v4_candidate.mjs` publishes only the exact validated seed-73 checksum
under `arkit-student-v4-candidate`, including the inherited licenses/notices and
a byte/hash manifest. Set `A2F_CDN_ROOT` to the existing CDN checkout and
`A2F_LICENSE_ROOT` to the archived upstream license directory. It reads existing
workspace CDN configuration without printing credentials, requires the staging
bucket/origin, and refuses mismatched existing objects instead of overwriting
them. Model files remain outside Git and the Vercel app bundle.

`verify_deployed_v4.mjs` accepts `DEPLOYMENT_URL` plus the Playwright/WebKit paths
above. It checks default WebGPU, v4 → v3 → v4, retained dialogue, model download
URLs, playback and ongoing idle poses. Set `TEST_SUPERTONIC=1` for an additional
hosted Supertonic generation. Use it against a staged production build before
promotion and the main URL afterward; this is desktop browser verification,
not physical-iPhone testing.

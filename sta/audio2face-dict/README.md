# Audio2Face dictionary studio

This SvelteKit 3 sandbox uses Phonon Easton and full TinyAligner in browser workers. The animation worker looks up recorded James ARKit motion by phoneme triple or double, retimes it, and blends joins. The audio element drives the 60 fps track during playback and seeking. The head motion slider controls the existing precomputed idle motion.

## Local setup

Use Node 22.18 or newer for the corpus tools. Prepare the James rig, James blendshape targets, and Phonon assets in the [Audio2Face studio](../../audio2face/README.md). The assets command copies the prepared files into this app’s ignored static directory. Set `JAMES_STUDIO_DIR` if the studio is elsewhere. The TinyAligner model and English lexicon come from the [full TinyAligner demo](../../model-demos/ttp/tinyaligner/README.md) or its pinned upstream revision.

```sh
npm ci
npm run assets
npm run dev
```

Open http://127.0.0.1:5189/. TinyAligner uses its English pronunciation dictionary and falls back to [Phonemize](https://github.com/hans00/phonemize) for unfamiliar words. Predicted pronunciations can differ from the audio, particularly for names. Spell out numbers. Open **Match coverage** after generation to see how much of the line used exact matches, single-phoneme fallback, or neutral motion.

The browser needs the generated dictionary in `static/dictionary/`. Build it with the commands below before using a fresh checkout. Run `npm run build` after preparing the assets and packing the desired corpus so the standalone Node build includes them.

## Generate and verify the dictionary

The labeler expects the existing full Audio2Face v3.2 Apple Silicon workflow, its verified model graph, and the matching James NVIDIA rig data. Use the workflow’s Python environment with NumPy and MLX. Set `A2F_PYTHON` to that Python executable, `A2F_RUNNER` to the workflow directory containing `app/audio2face_v3`, `A2F_MODEL` to its `network.onnx`, and `A2F_RIG` to the directory containing `bs_skin_James.npz`, `bs_skin_config_James.json`, and `model_data_James.npz`.

```sh
npm run corpus -- pilot
"$A2F_PYTHON" scripts/label.py .local/pilot --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG"
npm run pack -- pilot
npm test
node scripts/verify-pilot.mjs

npm run corpus -- full
"$A2F_PYTHON" scripts/label.py .local/full --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG"
npm run pack -- full
node scripts/evaluate.mjs
```

The pilot uses 26 training clips. The full job reuses the pilot and selects additional prompts for phone coverage until the synthesized training audio reaches 600 seconds. Pass a second argument in seconds to change that budget. The offline cache keeps mono 16 kHz PCM, full TinyAligner timings, and the fitted ARKit tracks. The packed browser dictionary contains the index and one Float32 animation track per training clip. The development and test clips are separate, and only training clips enter the browser bundle.

The evaluation command writes `.local/full/evaluation.json` for the full corpus and `.local/full/evaluation-targeted.json` for the targeted expansion. It measures match coverage and lookup time on the test set, and compares lip gaps against the direct ARKit fit on the same audio. Full geometry supplies a separate lip landmark reference for estimating the effect of the fit. A center lip contact is counted if the center gap reaches 0.1 James rig units within 20 ms of an aligned `/m/`, `/p/`, or `/b/` interval.

## Measure the targeted expansion

The targeted corpus adds 15 short training clips for five missing development pairs. The fresh set has 16 general sentences and eight sentences designed to exercise those pairs. The coverage script rejects evaluation transcripts already present in the corpus manifests and reports each group separately.

```sh
npm run corpus -- fresh
npm run pack -- full
node scripts/coverage.mjs

npm run corpus -- targeted
"$A2F_PYTHON" scripts/label.py .local/targeted --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG"
npm run pack -- targeted
node scripts/coverage.mjs
node scripts/coverage.mjs --fallback
```

The browser enables the fallback by default. When a pair is missing, it blends the outgoing half of the first phoneme with the incoming half of the next one. Where a neighboring exact match exists, the fallback uses its source phoneme to preserve the pose at the shared anchor. This uses the existing index and tracks. Coverage counts these intervals separately from exact triples and doubles.

## Compare fallback motion against full Audio2Face

With the targeted dictionary packed, label the fresh set with the same full Audio2Face runner and compare both playback variants on the same PCM.

```sh
"$A2F_PYTHON" scripts/label.py .local/fresh --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG" --geometry-reference
node scripts/compare-fallback.mjs
```

The report in `.local/fresh/fallback-quality.json` compares nine vertical lip gaps against both full geometry and the direct ARKit fit. It reports gap error across the clip and on fallback spans, error in frame-to-frame lip movement within 35 ms of a fallback boundary, and center lip contacts for `/m/`, `/p/`, and `/b/`. The closure comparison uses the 0.1 rig unit threshold and a 20 ms margin around each aligned phoneme. It lists the closures that changed when fallback was enabled.

The browser ranks up to eight candidates per exact triple or double and selects a path based on timing and compatibility of the mouth pose and motion at shared phonemes. Run `node scripts/compare-fallback.mjs --join-aware` to compare against timing-only selection on the same fresh set and write `.local/fresh/join-quality.json`. Both variants use the single-phoneme fallback.

The copied renderer and Phonon player come from the Audio2Face studio. Full TinyAligner feature extraction and CTC decoding follow the full demo and the upstream MIT implementation documented there. See the [implementation plan](docs/plans/audio2face-dict.plan.md).

# Audio2Face dictionary studio

This SvelteKit 3 sandbox uses Phonon Easton and full TinyAligner in browser workers. The animation worker looks up recorded James ARKit motion by phoneme triple or double, retimes it, and blends joins. The audio element drives the 60 fps track during playback and seeking. The head motion slider controls the existing precomputed idle motion.

## Local setup

Use Node 22.18 or newer for the corpus tools. Prepare the James rig, James blendshape targets, and Phonon assets in the [Audio2Face studio](../audio2face/README.md). The assets command copies the prepared files into this app’s ignored static directory. Set `JAMES_STUDIO_DIR` if the studio is elsewhere. The TinyAligner model and English lexicon come from the [full TinyAligner demo](../../model-demos/ttp/tinyaligner/README.md) or its pinned upstream revision.

```sh
npm ci
npm run assets
npm run dev
```

Open the URL printed by Vite. Development starts at port 5173 and preview at 4173, advancing to the next available port when needed. TinyAligner uses its English pronunciation dictionary and falls back to [Phonemize](https://github.com/hans00/phonemize) for unfamiliar words. Predicted pronunciations can differ from the audio, particularly for names. Spell out numbers. Open **Match coverage** after generation to see how much of the line used exact matches, single-phoneme fallback, or neutral motion.

Use **Prev** and **Next** beside **Dialogue** to cycle through twenty editable manual test samples. They include introductions, tongue twisters, short Shakespeare excerpts, a public-domain *Animal Crackers* monologue, two original cinematic speeches, and tests for closures, rounding, consonants, pauses, connected speech, and unfamiliar names. Speech excerpts target roughly thirty seconds; actual duration depends on synthesis. Choosing a sample clears the previous playback and diagnostics; press **Generate & play** to run it. These samples are for manual inspection, separate from the fixed evaluation sets. The film excerpt is sourced from [Filmsite](https://www.filmsite.org/anim2.html).

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

The pilot uses 26 training clips. The full job reuses the pilot and selects additional prompts for phone coverage until the synthesized training audio reaches 600 seconds. Pass a second argument in seconds to change that budget. The offline cache keeps mono 16 kHz PCM, full TinyAligner timings, and the fitted ARKit tracks. The default fit minimizes both face vertex error and the vertical gap between nine pairs of lip landmarks. The packed browser dictionary contains the index and one Float32 animation track per training clip. The development and test clips are separate, and only training clips enter the browser bundle.

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

The browser ranks up to eight candidates per exact triple or double and selects a path based on timing and compatibility of the mouth pose and motion at shared phonemes. The lip fit also compares the predicted opening at the center and both sides of the lips when scoring joins. Run `node scripts/compare-fallback.mjs --join-aware` to compare against timing-only selection on the same fresh set and write `.local/fresh/join-quality-lip.json`. Both variants use the single-phoneme fallback.

To reproduce the original vertex fit, pass `--fit baseline` to both `scripts/label.py` and `npm run pack -- targeted`. It uses a separate cache and writes the join comparison to `.local/fresh/join-quality.json`. The two reports measure dictionary playback and the direct ARKit fits against the same full Audio2Face lip landmarks.

## Train and evaluate closure guidance

The targeted dictionary can use a small acoustic model to rank and blend poses for `/m/`, `/p/`, and `/b/`. Train it on the two training corpora using lip landmarks from full Audio2Face. The packer includes the trained model when present. To build it from an existing cache, run:

```sh
"$A2F_PYTHON" scripts/label.py .local/full --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG" --geometry-reference
"$A2F_PYTHON" scripts/label.py .local/targeted --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG" --geometry-reference
node scripts/train-closures.mjs
npm run pack -- targeted
node scripts/compare-fallback.mjs --closures
node scripts/compare-fallback.mjs --closures --test-set
```

The two reports compare the same packed tracks with closure guidance on and off. A recorded contact pose can be reused only when the audio strongly supports closure, the current motion is already close to contact, and the source pose shares a neighboring phoneme. The model and corpus data are generated locally.

The copied renderer and Phonon player come from the Audio2Face studio. Full TinyAligner feature extraction and CTC decoding follow the full demo and the upstream MIT implementation documented there. See the [implementation plan](docs/plans/audio2face-dict.plan.md).

## Select subtitle prompts for phoneme pair coverage

Scan a local English subtitle text file with TinyAligner’s stress-free pronunciation dictionary.

```sh
node scripts/analyze-subtitles.mjs /path/to/OpenSubtitles.en-fr.en
node scripts/verify-subtitles.mjs .local/subtitle-analysis/augmentation.json
```

The scan streams the corpus and counts ordered pairs within words and across word boundaries. Unknown words and numeric tokens break the sequence. Candidate lines use known words, and evaluation transcripts from this demo are excluded. A bounded candidate pool feeds a greedy selection weighted by pair frequency and sentence length. The outputs in `.local/subtitle-analysis/` include frequencies, a standalone selection, and an augmentation for the current dictionary.

The verifier synthesizes the selected text with Phonon, runs full TinyAligner, and confirms pairs using the same pause detection as playback. Its coverage estimate uses the corpus pair frequencies. Prepare and label those verified clips before packing them as animation samples:

```sh
node scripts/prepare-subtitles.mjs
"$A2F_PYTHON" scripts/label.py .local/subtitles --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG"

# Save the previous dictionary for a comparison with identical settings.
npm run pack -- targeted
mkdir -p .local/subtitle-baseline
cp static/dictionary/index.json static/dictionary/weights-*.f32 .local/subtitle-baseline/
npm run pack -- subtitles
node scripts/compare-fallback.mjs --baseline .local/subtitle-baseline --output .local/fresh/subtitle-quality-lip.json
node scripts/compare-fallback.mjs --baseline .local/subtitle-baseline --test-set --output .local/full/subtitle-quality-lip.json
node scripts/coverage.mjs --fallback
node scripts/evaluate.mjs
```

The subtitle mode combines the full, targeted, and subtitle training corpora. It preserves the closure model, lip fit, and playback settings for a comparison of the recordings alone. Preparation rejects duplicate transcripts and overlap with the existing evaluation sets. Comparisons use the same cached speech, alignments, and full Audio2Face geometry references. Both versions are measured at the union of their join anchors, within 35 ms of each anchor.

The first expansion adds 250 clips (849.84 seconds) to the existing 254 clips (630.64 seconds). The packed dictionary contains 504 clips, 1,230 pair keys and 4,958 triple keys, totaling 21,114,463 uncompressed bytes for the index and active animation weights. The original clips and animation tracks are preserved exactly.

| Evaluation set | Exact interval coverage, before → after | Lip gap RMS vs full geometry | Join step error RMS |
| --- | --- | --- | --- |
| Same 24 fresh sentences | 91.73% → 100% | 0.39149 → 0.36465 | 0.14104 → 0.12838 |
| Same 12 test sentences | 93.21% → 99.62% | 0.33689 → 0.30375 | 0.14448 → 0.12655 |

Closure results are mixed at the unchanged 0.1 rig-unit threshold. On the fresh set, correct `/m/`, `/p/`, `/b/` contacts change from 7/13 to 6/13, while extra contacts change from 3 to 1. On the test set, correct contacts change from 9/14 to 10/14, while extra contacts change from 2 to 3. These are contact detections inside the evaluated phoneme windows, not a count of every contact in the clip. Larger pair coverage improves the overall gap and join metrics here but does not guarantee every closure improves. The measurements are numerical; no claim of subjective visual improvement is implied.

## Experimental acoustic selection and closure retiming

These options are **disabled in normal playback** because the fixed-set comparison showed a tradeoff rather than an overall improvement. The recordings, ARKit tracks, closure classifier, and exact match coverage stay fixed.

With `--acoustics`, the packer stores eight small descriptors per phoneme: gain-normalized energy and brightness at three phases, relative pitch, and voicing strength. Normal packs omit this experimental data. The optional ridge model estimates three lip openings from these descriptors, phone duration, and adjacent phone identities. It trains only on the packed training clips and their recorded ARKit poses. These estimates rank recorded clips; they do not generate animation. The model is bound to the packed animation weights and is included only when that fingerprint matches.

The retiming option uses strong acoustic closure evidence to move a nearby recorded contact by at most forty milliseconds. The local time map stays monotonic, has slopes between one half and two, and keeps its support endpoints fixed. It cannot create a closure pose absent from the recording. Existing closure blending and conservative contact restoration are unchanged.

```sh
# With the 504-clip subtitles corpus already packed:
npm run pack -- subtitles --acoustics
"$A2F_PYTHON" scripts/train-selection.py
npm run pack -- subtitles --acoustics
node scripts/tune-acoustics.mjs
node scripts/compare-fallback.mjs --optimization --output .local/fresh/selection-quality-lip.json
node scripts/compare-fallback.mjs --optimization --test-set --output .local/full/selection-quality-lip.json
# Restore the normal bundle after the experiment.
npm run pack -- subtitles
```

The tuning sweep uses only the seven development sentences. The combined comparison uses acoustic weight 0.25, selected from that sweep. Add `--acoustic-only` or `--retiming-only` for an ablation; `--acoustic-mode similarity` selects the raw descriptor distance experiment. Both playback defaults remain off unless `planAnimation` receives `acousticAware: true` and/or `closureRetiming: true`.

| Set | Lip gap RMS, current → combined | Join step error RMS, current → combined | Correct / extra closures, current → combined |
| --- | --- | --- | --- |
| 24 fresh sentences | 0.36465 → 0.36381 | 0.13060 → 0.12861 | 6 / 1 → 6 / 1 |
| 12 test sentences | 0.30375 → 0.30589 | 0.12808 → 0.12446 | 10 / 3 → 10 / 3 |

The combined candidate reduces join error by 1.5% and 2.8%, respectively, but increases test-set lip gap error by 0.7%. Contact totals are unchanged, although two false contact detections exchange places on the test set. The raw descriptor experiment was also rejected: it increased test-set error and introduced an additional false contact. Retiming alone had negligible effects. These results do not justify enabling the new defaults yet.

## P/B approximate triple fallback

Playback enables a conservative, symmetric `P ↔ B` triple fallback. Exact triples keep priority. Where an interval would otherwise need doubles or singles, and no exact triple starts there, the matcher can replace one `P` with `B`, or one `B` with `P`, at any of the three positions. It never substitutes `M`, changes two phones, or crosses confirmed silence. The original transcript, aligned phone labels, and target timings remain unchanged.

Approximate triples compete with exact doubles in the join-aware path search. The timing score includes a substitution penalty, larger for the central phone, and is normalized by contributed intervals. Uncovered intervals are penalized so an exact double cannot win merely by leaving the next interval unsupported. The path scorer supports both lengths and compares their overlapping mouth motion. Exact triple candidates themselves stay unchanged; an approximate fallback is not started on an interval already covered by an earlier exact triple.

The UI reports exact triples, approximate triples, doubles, and single sources separately. The expanded coverage panel counts an interval as approximate whenever any contributing clip is approximate, even if another contributing clip is exact. Counts therefore distinguish source matches from blended intervals. Use `visemeFallback: false` with `planAnimation` to reproduce the previous algorithm. The acoustic and retiming experiments remain disabled.

```sh
node scripts/compare-fallback.mjs --viseme-fallback --output .local/fresh/pb-quality-lip.json
node scripts/compare-fallback.mjs --viseme-fallback --test-set --output .local/full/pb-quality-lip.json
```

Both versions use the same 504 recordings, PCM, alignments, closure model, and full Audio2Face references. The fixed 24-sentence set selects no approximate triples and is unchanged. The 12-sentence set selects one approximate triple and replaces one exact double: `B M UW` uses `P M UW` in “Bob moved the book beside the lamp.” This contributes to two blended intervals and restores the lip closure on the initial `B` at 0.42 seconds. `P → B` and `B → P` are both covered by routing tests at all three positions; the fixed sentences exercise only the latter direction in selected motion.

| Set | Selected approximate triples | Lip gap RMS, before → after | Join step error RMS, before → after | Correct / extra M/P/B closures |
| --- | --- | --- | --- | --- |
| Same 24 fresh sentences | 0 | 0.36465 → 0.36465 | 0.13060 → 0.13060 | 6 / 1 → 6 / 1 |
| Same 12 test sentences | 1 | 0.30375 → 0.30251 | 0.12791 → 0.12754 | 10 / 3 → 11 / 3 |

Whole-utterance contact episodes provide an additional check for closures outside M/P/B windows. With the same gap threshold and a twenty-millisecond matching tolerance, correct / extra episodes remain 45 / 26 on the fresh set and change from 34 / 10 to 35 / 10 on the test set. Join measurements use identical windows for both variants, taken from the union of their join anchors; join values from comparisons with different anchor sets should not be compared directly. Evidence of improvement is limited to the one selected approximate match.

## Measure closure offsets from training speech

Generate native lip gap references for all training sources, then measure contact episodes independently of the narrow phoneme evaluation window:

```sh
for corpus in full targeted subtitles; do
  "$A2F_PYTHON" scripts/label.py ".local/$corpus" --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG" --geometry-reference
done
node scripts/analyze-closure-offsets.mjs
```

The analyzer reads only `split: train` rows: 504 clips, 1,480.48 seconds and 1,174 P/B/M occurrences. It detects contiguous center-lip contact at gap <= 0.1, interpolating threshold crossings between the 60 fps samples. It associates episodes within 200 ms of each phoneme, but includes a case in primary statistics only when there is exactly one eligible episode and no other P/B/M eligible for it. Shared/neighbor ambiguities, multiple episodes, absent contacts and clip-truncated contacts have separate counts. These exclusions produce 513 unambiguous native-geometry examples, rather than assuming every nearby closure belongs to the closest phoneme.

| Phone | Native examples | Median onset relative to phone start | Central 90% onset range | Median release relative to phone end | Central 90% release range |
| --- | --- | --- | --- | --- | --- |
| P | 203 | -84 ms | -158 to -52 ms | -45 ms | -75 to -9 ms |
| B | 176 | -102 ms | -162 to -62 ms | -49 ms | -73 to -25 ms |
| M | 134 | -77 ms | -144 to -31 ms | -48 ms | -76 to -1 ms |

Negative values mean contact or reopening precedes the corresponding acoustic boundary. Median contact durations are 69 ms for P, 76 ms for B and 58 ms for M. Contact midpoint offsets relative to phoneme midpoint have medians of -67, -76 and -63 ms, respectively. The stored ARKit fit shows similar median onsets (-82, -101 and -75 ms); on the paired unambiguous subset, fitting delays onset by about three milliseconds and advances release by about three to five milliseconds.

Of the 513 unambiguous native contacts, 287 (56%) end before the old phoneme-start-minus-20-ms window even begins. This confirms that a miss in that window is not sufficient evidence of absent lip closure. The threshold/window sensitivity report covers contact thresholds 0.05, 0.10 and 0.15 and association margins 150, 200 and 250 ms; onset medians remain similar. The output in `.local/closure-offsets/` contains summary and corpus-specific distributions, sensitivity results, individual associations and a CSV of offsets.

These are descriptive offsets for this Phonon Easton, TinyAligner and Audio2Face pipeline. They include any systematic alignment/model timing difference and do not isolate human anticipatory articulation. They are not universal phoneme constants or a justification for a global time shift. The reference resolution is 16.7 ms and TinyAligner labels are on a 20 ms grid; interpolated decimal values should not be interpreted as sub-millisecond timing accuracy. No playback timing or evaluation thresholds are changed by this analysis.

## Evaluate closures in their learned advance windows

Closure evaluation now uses the training-only profile in `data/closure-windows.json`. Regenerate it deliberately after recalibration:

```sh
node scripts/analyze-closure-offsets.mjs --write-profile
node scripts/compare-fallback.mjs --viseme-fallback --output .local/fresh/pb-advance-quality.json
node scripts/compare-fallback.mjs --viseme-fallback --test-set --output .local/full/pb-advance-quality.json
node scripts/evaluate.mjs
```

The profile records the native training contact count, percentile bounds, pipeline description, threshold, and a hash of the training clip IDs, audio lengths and phoneme timings. Evaluation never refits the profile to the evaluation sentences. A phone's search window runs from its start plus the fifth-percentile onset offset to its end plus the ninety-fifth-percentile release offset. Contacts are detected as episodes, not isolated low-gap frames.

`advanceWindowClosures` is the primary metric in both evaluation tools. A reproduced contact means both the candidate and reference have one unambiguous episode in that phone's advance window; it does not mean their timing is perfect. Onset, release, duration, and minimum-gap errors are measured against the actual full Audio2Face event. Contacts outside the training onset/release percentile bands are flagged as timing outliers rather than automatically discarded. Multiple contacts, potentially shared contacts, and clipped windows/events are explicitly unscored, rather than counted as successes or credited to more than one phone. The comparison report includes per-phone events and classifications for inspection.

Whole-utterance episode matching remains a separate safeguard for extra and missing contacts, including events outside the learned phone windows. The former +/-20 ms counts remain in the JSON for historical comparisons, labelled as legacy in settings; they are no longer the primary closure check. Playback and the packed dictionary are unchanged.

With the current P/B fallback enabled:

| Set | Reproduced contacts | Missing contacts | Extra contacts in scored phone windows | Correctly absent | Unscored |
| --- | --- | --- | --- | --- | --- |
| Same 24 fresh sentences (42 P/B/M occurrences) | 18 | 11 | 2 | 4 | 7 |
| Same 12 test sentences (40 P/B/M occurrences) | 9 | 9 | 1 | 0 | 21 |

On matched contacts, mean absolute onset/release errors are 30.8/30.5 ms and 23.5/17.2 ms, respectively. Of the unscored cases, the native reference is ambiguous or clipped in 5 and 18 cases, and the candidate has an ambiguous or clipped episode in a further 2 and 3 cases. The P/B-off comparison has the same learned-window counts; the previously recovered `B` in “Bob moved…” shares a reference episode with the following `M`, so per-phone attribution is explicitly unscored here. Whole-speech event matching still captures that improvement. Wider advance windows reveal many real reference contacts that the old metric missed, so old and new denominators should not be compared as a quality change.

## Advance-window selection and blending

The default playback guide now predicts contacts inside the learned advance windows. Its separate, small model in `data/advance-closure-model.json` is trained on 14,669 frames from the same 504 training clips and their full Audio2Face geometry. It uses acoustic features and an absolute offset from phoneme start, avoiding the inappropriate duration-scaled phase of a very short acoustic alignment. Its training manifest/timing hash must match the timing profile. Evaluation clips are never included in fitting.

The guide ranks candidates only at times they actually contribute, including advance contacts belonging to the following phoneme. Confident predicted contact can bias a blend toward a donor already in contact; two open donor poses cannot be made closed by that blend. Conservative recorded-pose restoration uses advance windows for both target and donor. A local retention pass restores baseline contact motion, with a short crossfade, if reranking loses an existing event in an advance window. It handles shared events once and does not inspect evaluation references at runtime. The original dictionary, P/B fallback and phoneme timings are unchanged. The previous acoustic descriptor matching and nonlinear retiming experiments remain off.

```sh
node scripts/train-advance-closures.mjs
node scripts/compare-fallback.mjs --advance-closures --output .local/fresh/advance-motion-quality.json
node scripts/compare-fallback.mjs --advance-closures --test-set --output .local/full/advance-motion-quality.json
```

Use `advanceClosures: false` in `planAnimation` to reproduce the previous playback. Both comparison variants retain the P/B fallback and use identical audio, dictionary tracks and full Audio2Face references. Whole-utterance contact episodes are the primary closure metric for this comparison, so shared closures count once. The per-phoneme advance-window checks remain available, including all ambiguous cases. Reports include per-sentence matched, extra and missing event intervals for auditing.

| Fixed set | Reproduced whole-speech contacts | Missing contacts | Extra contacts | Lip gap RMS | Join step error RMS |
| --- | --- | --- | --- | --- | --- |
| 24 fresh sentences | 45 → 46 of 62 | 17 → 16 | 26 → 26 | 0.364652 → 0.364628 | 0.130601 → 0.129415 |
| 12 test sentences | 35 → 36 of 46 | 11 → 10 | 10 → 10 | 0.302511 → 0.302649 | 0.127620 → 0.127994 |

The recovered reference contacts are the early B in “brush” (0.333–0.417 s) and “beach” (1.333–1.383 s). An event-level regression check confirms every previously reproduced reference contact remains reproduced and each candidate extra event overlaps an existing baseline extra event. This is evidence of two recovered contacts on these sets, not a claim that extra closures are eliminated for arbitrary speech. Overall lip error is nearly unchanged; fresh-set join error improves 0.91%, while test-set lip and join errors increase 0.046% and 0.29%. Mean selection time in the Node comparison increases from about 1.1–1.8 ms to 1.6–2.6 ms, including the baseline retention plan. The browser dictionary remains 21.11 MB; the guide model adds about 3 KB to the application bundle before compression.

## Preserve recorded opening context

The “brave” onset issue came from discarding the selected recording's opening context. The James neutral pose is closed, and the selected source vowel's acoustic start was already in the anticipatory closure for the following B. Fading from neutral directly to that pose kept the lips closed continuously from frame zero, hiding a distinct B onset.

When an utterance starts with a vowel followed by P/B/M, and the selected vowel onset is already closed, playback now looks backward in that same recording for a real open-to-closed lead-in. It uses only the final contiguous open run before that closure, bounded to 200 ms and at most twice the target leading duration. That prevents an unrelated earlier donor closure from leaking into the track. The first animation frame is initialized from the recorded lead-in pose rather than from zero weights. It leaves the original matching, phoneme timings, and motion from the first phoneme onward unchanged. If there is no usable recorded opening or no target lead time, it makes no opening up. Initial consonants and already-open source vowels retain their existing behavior.

```sh
node scripts/compare-fallback.mjs --opening-context --output .local/fresh/opening-quality.json
node scripts/compare-fallback.mjs --opening-context --test-set --output .local/full/opening-quality.json
```

Both variants keep advance guidance and P/B fallback enabled. `openingContext: false` reproduces the version before this fix. In “A brave child crossed the muddy field,” the selected donor contributes 0.667–0.860 s of recorded context to the target's 0–0.160 s lead-in. The assembled B now closes at 144.6 ms, compared with the full Audio2Face onset at 160.0 ms, and reopens at 273.7 ms (unchanged; reference 304.7 ms). It is now a complete, matched episode instead of an onset clipped by frame zero. Its onset is about 15 ms early, so this is not a claim of exact timing; it remains flagged as an onset percentile outlier in the per-phone report.

The fixed comparison changes only this sentence. Its lip gap RMS falls from 0.40735 to 0.27821 (31.7%). Across all 24 fresh sentences RMS falls from 0.364628 to 0.358713 (1.62%); join error is unchanged. Whole-speech matched/extra/missing counts remain 46/26/16, with all previously matched reference events retained and no new extra episodes. The B is now also scored as reproduced in the advance-window check instead of excluded as clipped. All output frames in the 12-sentence test set remain unchanged, with counts 36/10/10. The dictionary and audio are unchanged.

## Measure context for rounding, jaw opening and F/V positioning

The analysis tools measure native geometry and the corresponding stored ARKit fit on all 504 training clips. They do not include development/test sentences or change playback.

```sh
"$A2F_PYTHON" scripts/extract-motion-features.py --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG"
node scripts/analyze-phoneme-context.mjs --write-candidates
"$A2F_PYTHON" -m unittest discover -s tests -p test_motion_features.py
```

Native inference records lip narrowing, fixed-head lip protrusion, downward jaw-marker displacement, and lower-lip proximity to a dense upper-incisor vertex patch. Corresponding fitted measurements use the same lip landmarks, skin rig and packaged jaw targets. Protrusion uses the fixed head reference: the mouth corners move forward in a pucker, so using them as the reference would incorrectly cancel rounding. F/V proximity is unsigned nearest-vertex distance from three central lower-lip markers, not proof of physical lip/tooth surface contact.

Events are continuous periods above native training feature thresholds. Rounding requires both narrowing and protrusion; R is included as an ownership probe so its neighboring motion is not silently credited to a rounded vowel. The main threshold is the training 75th percentile, with 65th/85th percentile sensitivity checks. An event must be complete and uniquely associated with one eligible phoneme within the search neighborhood to enter primary statistics. Shared, multiple, absent and truncated events are counted separately. Threshold crossings for continuous motion are operational definitions, not unique physiological boundaries.

| Feature / phoneme | Native unambiguous examples | Median feature onset relative to phoneme start | Median feature release relative to phoneme end |
| --- | --- | --- | --- |
| Rounding / W | 52 | -126 ms | -31 ms |
| Rounding / UW | 56 | -87 ms | +12 ms |
| Rounding / OW | 64 | -90 ms | +71 ms |
| Rounding / R probe | 126 | -123 ms | -46 ms |
| Lip-to-incisor proximity / F | 165 | -122 ms | +28 ms |
| Lip-to-incisor proximity / V | 116 | -97 ms | +1 ms |
| Strong jaw opening / EY | 35 | -27 ms | +10 ms |

Positive release offsets mean the measured shape persists after the acoustic interval. Timing is sampled at 60 fps against 20 ms aligner timestamps; rounded decimal differences are not sub-millisecond timing accuracy. These are model/voice/aligner-specific associations, not universal human phoneme constants.

Most strong jaw-opening events cannot be attributed to one vowel: 4,363 of 5,881 eligible occurrences are shared/ambiguous under this conservative rule. Only EY reaches the provisional minimum of thirty unambiguous jaw examples; this does not mean other vowels have no jaw motion. The other rounded-vowel probes also lack enough uniquely attributable examples for individual ranges.

`data/phoneme-context-candidates.json` covers all 39 observed phonemes. Ten have provisional margins from these features or the existing P/B/M closure data; unsupported entries are null. Every entry is inactive. Candidate margins span the P05-onset/P95-release envelope across feature thresholds, plus one frame of context, and are intentionally wider than median feature timings. Broad tails and threshold-sensitive candidates are flagged. Some proposed margins exceed 250 ms because continuous shapes can persist through neighboring sounds; those are not safe unconditional blend extensions. Donor-event boundaries, adjacent context and silence still need to constrain them in playback experiments.

The complete distributions, sensitivity checks, individual events and CSV are in `.local/phoneme-context/`. The extraction cache records the model, training manifest and measurement-code fingerprints; the analyzer rejects a stale measurement definition. These profiles must be benchmarked against the current playback before any before/after blend margins are enabled.

## Bounded feature-context blending experiment

`planAnimation(dictionary, phones, pcm, { boundedContext: true })` enables measured before/after context around selected double and triple clips. It uses the per-phone margins in `data/phoneme-context-candidates.json`, capped by adjacent source and target midpoints, speech-run/silence boundaries, available donor frames and output duration. Unsupported phones get no added context. The dedicated utterance-opening fix is retained. This pass changes blending only; selection and coverage stay the same.

Each contribution uses a smooth sine-squared taper with zero weight at both ends, default strength 0.25, source speed bounded to 0.5–2 and accumulated per-channel blend weight capped at 0.5. Threshold-sensitive and broad profiles receive lower weights. Only channels relevant to the measured feature are affected: rounding, jaw opening, F/V positioning or closure. Donor motion cannot cross a silence boundary. A geometric contact guard attenuates or rejects changes that would flip the current center-lip contact state at any 60 fps frame, including a Float32 boundary safeguard. This preserves existing contact episodes; it cannot recover missing ones or remove existing extras. It does not guarantee unchanged sub-frame crossings or all lip geometry on other avatars.

Reproduce the comparison against current playback, using the same audio, dictionary, selected clips and full Audio2Face references:

```sh
node scripts/compare-fallback.mjs --bounded-context --development-set --output .local/bounded-development.json
node scripts/compare-fallback.mjs --bounded-context --output .local/fresh/bounded-context-quality.json
node scripts/compare-fallback.mjs --bounded-context --test-set --output .local/full/bounded-context-quality.json
```

The comparison explicitly enables the current P/B fallback, advance closure guidance and opening-context fix in both variants. Both variants disable the older acoustic and nonlinear-retiming experiments. `--context-strength` accepts a value greater than zero and no larger than 0.5. Reports include actual source/target context ranges and contact-limited frame counts; the harness asserts identical contact masks frame by frame.

The initial absolute-pose experiment at frozen strength 0.25 changed fresh-set lip-gap RMS from 0.358713 to 0.359137 (+0.12%) and join error from 0.129415 to 0.129131 (-0.22%). The 12 test sentences changed lip-gap RMS from 0.302649 to 0.301977 (-0.22%) and join error from 0.127994 to 0.127477 (-0.40%). Matched/extra/missing contacts stayed exactly 46/26/16 and 36/10/10 respectively, with every original event retained. The seven development sentences regressed slightly on both motion and join error.

### Investigation and revised blending

Isolating each phoneme's old windows on the fresh sentences identifies EY jaw-opening context as the largest contributor: its windows alone increase lip-gap RMS by about 0.000471, versus the combined increase of 0.000424 (individual effects need not add linearly). The worst affected sentence is “They watched the blue whale rise.” The EY profile contains only 35 unambiguous jaw events, 11.9% of eligible occurrences. The initial blend used that weakly attributable, general jaw feature at full confidence. It also imported absolute donor poses and allowed a different neighboring phoneme outside the matched n-gram: 127 of 186 old fresh-set context regions have incompatible outer neighbors, including seven of thirteen EY regions. Time bounds alone do not establish that a donor gesture belongs to the target context.

The revised `contextBlend: 'anchored'` mode (the default when explicitly enabling bounded blending) aligns every donor feature pose to the existing assembled pose at its phoneme midpoint. It transfers surrounding motion rather than the donor's absolute pose bias; all offsets are computed before any frame is changed, preventing feedback between overlapping contexts. Aligned donor coefficients remain clamped to [0,1]. Context influence is multiplied by the smaller source/target training attribution fraction when available. Outside neighbors must match exactly, allowing symmetric P/B equivalence, and the source time slope uses the outside interval being sampled. All prior bounds, tapering and contact guards remain in place. At run ends without an outside neighbor, it retains the interior slope and the existing tail bounds.

Use `--context-blend absolute` to reproduce the initial experiment and `--context-blend anchored` for the revision. `--context-phones EY` (or a comma-separated list) isolates a phoneme's context for diagnosis; it does not change selection. The revised design was checked on the seven development sentences before rerunning the fixed fresh/test comparisons; attribution weights come from training measurements, not full Audio2Face reference poses for those sentences.

| Geometry metric | Current playback | Initial windows | Revised windows |
| --- | ---: | ---: | ---: |
| Fresh 24 lip-gap RMS | 0.358713 | 0.359137 | 0.358701 |
| Fresh 24 join error | 0.129415 | 0.129131 | 0.129438 |
| Test 12 lip-gap RMS | 0.302649 | 0.301977 | 0.302636 |
| Test 12 join error | 0.127994 | 0.127477 | 0.127982 |

The revised pass admits 61 context regions on fresh sentences (previously 186) and 42 on test sentences (previously 115). It removes the fresh-set shape regression, but the remaining shape gains versus current playback are only 0.0034% and 0.0046%, and it loses most of the initial join gains. Every center-contact frame, matched event, existing extra event and missed reference event remains identical on both sets. This is a conservative reduction of harmful context transfer, not a demonstrated meaningful fidelity improvement. These are geometric measurements against full Audio2Face, not a visual-quality verdict. The experiment remains **off by default** (`profile.active: false`); normal browser playback retains the current version.

## Recorded internal pauses: opt-in pilot

**Recorded pauses (pilot)** is checked by default in the browser, using recorded release/rest/preparation motion where supported. Uncheck it to use synthetic relaxation throughout. Match coverage details report how many detected pauses used recordings. API callers can pass `{ recordedPauses: true }` to `planAnimation`; single-phoneme, join and closure selection remain unchanged. The bounded feature-context experiment stays off independently. The initial pilot comparison below retains the historical 25 ms fade in both variants; current playback now uses the slower fallback described in the next section.

The training bundle contains 215 internal quiet intervals (189 at least 200 ms), with full ARKit motion already present in the existing weight file. Packing now retains the quiet timestamps alongside speech-run boundaries. Rebuild an older bundle with `npm run pack -- subtitles`; no Audio2Face inference or new corpus clips are needed. Older bundles without timestamps gracefully retain current playback. This metadata change leaves the dictionary keys, clip order, weights and closure model unchanged.

Silence alone does not establish a useful rest pose. The pilot searches recorded quiet frames for low mouth activity and near-rest lip separation, excludes the next P/B/M's learned advance window (or the last 40 ms before other sounds), and rejects large unrelated opening excursions in release/preparation paths. Seventy-five of the 215 recorded intervals provide a qualified rest anchor. Outgoing and incoming donors must match the respective boundary phoneme; matching two-phoneme context is preferred. Both sides must be usable, otherwise the pause retains its existing playback. This currently excludes, for example, the P→pause boundary in the existing development sentence “Bring the cup. Then shut the door.”

Each accepted pause plays a recorded release towards its donor's rest frame and a recorded preparation away from the incoming donor's rest frame. It holds those recorded rests when needed, bounds source speed to 0.5–2 and crossfades with smoothstep. Short endpoint corrections align donor poses to the existing speech at both boundaries. Only mouth/jaw channels change, all coefficients stay within [0,1], and frames outside the pause are bit-identical. The center-lip contact guard preserves every original speech-contact frame, including P/B/M advance windows and one frame of margin, with a short easing region outside protected speech. A transition that adds an isolated quiet contact cycle is reverted in full. This avoids confusing ordinary resting contact with an extra speech closure.

The original 24 fresh and 12 test sentences have no detected internal pauses and remain bit-identical with the pilot enabled. For an effective pause comparison, `scripts/prepare-pause-pilot.mjs` pairs the same cached recordings, retaining 60 ms after the first alignment and 120 ms before the second, with intended boundary gaps of 200/350/500 ms. Timings are shifted directly, without changing their phonemes or regenerating TTS. There are twelve fresh pairs, six test pairs and three development pairs. One intended short test gap does not pass the existing acoustic pause detector, leaving five detected test pauses. Full native Audio2Face references are inferred on each complete paired waveform, not spliced from separate tracks. The paired test design controls audio and timing but does not reproduce naturally synthesized multi-sentence prosody.

```sh
node scripts/prepare-pause-pilot.mjs
for split in development fresh test; do
  "$A2F_PYTHON" scripts/label.py ".local/pause-pilot/$split" --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG" --geometry-reference
done
node scripts/compare-pauses.mjs --all
```

The comparison writes `.local/pause-pilot-comparison.json` and review tracks in `.local/pause-review/`. Without `--all`, it uses only development recordings. Metrics compare nine lip gaps with full native geometry; transition steps and accelerations are measured across each whole pause and a 50 ms border. Results include unchanged/rejected pauses, not just accepted cases.

| Paired held-out recordings | Fresh: 12 pairs | Test: 6 pairs |
| --- | ---: | ---: |
| Pauses accepted / detected | 5 / 12 | 2 / 5 |
| Pause lip-gap RMS, current → pilot | 0.795755 → 0.763650 | 0.553474 → 0.487443 |
| Transition step-error RMS, current → pilot | 0.195912 → 0.183637 | 0.181862 → 0.163984 |
| Transition acceleration-error change | -3.7% | -0.5% |
| Exact zero-weight pause frames, current → pilot | 198 → 107 | 95 → 51 |
| Speech contact episodes, current → pilot | 35 → 35 | 32 → 32 |

All protected speech-contact masks and speech-run episode masks are identical before/after. No accepted transition adds an isolated quiet-contact cycle; one proposed fresh transition is rejected for doing so. Ordinary quiet-only contact episodes fall by two in the fresh pairs and one in the test pairs. Three development pairs improve pause shape and transition step errors by 3.6% and 5.1%, accepting one of four detected pauses. Threshold/source filters were developed on those cases; a unit-test fix to the non-labial timing default was then included in the final comparisons. The worst transition peak is unchanged on both held-out groups because unsupported pauses still use the old fade. These results support an opt-in pilot, not universal pause coverage or a general visual-quality claim.

Side-by-side rendered mouth reviews compare current playback, the pilot and the full Audio2Face ARKit fit on identical audio. They show a gentler release and a nonzero resting pose in accepted examples. The reference rendering is the bounded ARKit fit, while the numerical reference is native geometry. End-to-end browser generation with the pilot enabled was also checked, including the reported pause count and playback.

## Slower neutral transitions for remaining pauses and speech endings

Normal playback now relaxes over **up to 200 ms** using a quintic ease with flat velocity and acceleration at each end of the envelope. Internal pauses cap each outgoing/incoming fade at half the detected quiet interval, so the fades do not overlap. A final tail uses up to 200 ms, capped to the available trailing audio. For example, 120 ms of internal quiet gives 60 ms on each side; 540 ms gives two 200 ms fades and 140 ms at neutral. The measured audio duration and speech timestamps are unchanged.

This runs after normal sampling, closure restoration and the recorded-pause pilot. It uses the actual final/next baked speech poses, including closure corrections, instead of reverting to uncorrected donor endpoints. Speech frames, initial preparation, uncovered voiced dictionary intervals, and accepted recorded-pause intervals are left unchanged. Supported recorded pauses retain their recorded rests. The slower synthetic fade handles unsupported/rejected pauses and utterance endings whether the pilot checkbox is on or off. `{ smoothNeutral: false }` reproduces the previous fade; `{ neutralFadeSeconds: .2 }` is the default duration.

Speech-contact masks are protected over actual phoneme intervals and the learned anticipatory P/B/M windows, with one frame of advance margin. Incoming protection is eased in over 80 ms; it is not carried forward beyond a completed sound. Resting contact can therefore occur later, rather than retaining the old 25 ms resting-contact timing. Every interval is checked against its original center-lip contact-episode count; if it adds an episode, that interval is restored. This fallback is relevant for unusually truncated tails; none of the measured intervals needed it. Rest-contact timing and some redundant resting episodes may change, while speech closures stay intact. Coefficients remain in [0,1], and no extra animation duration is appended.

Reproduce comparisons on the original 24 fresh / 12 test recordings and their paired-pause versions:

```sh
node scripts/compare-pauses.mjs --smooth-neutral --all
node scripts/compare-pauses.mjs --smooth-neutral --with-recorded-pauses --all
```

The first writes `.local/neutral-transition-comparison.json`; the second writes `.local/neutral-transition-comparison-recorded.json` and keeps the recorded-pause pilot enabled in both variants. Review tracks are under `.local/neutral-review/` and `.local/neutral-review-recorded/`. Historical feature comparisons and the original recorded-pause comparison explicitly disable this newer transition to preserve their original baselines.

| Metric versus full native Audio2Face | Fresh | Test |
| --- | ---: | ---: |
| Original recordings: tail lip-gap RMS | 0.241400 → 0.231936 (-3.9%) | 0.229629 → 0.186815 (-18.6%) |
| Original recordings: end-transition step-error RMS | 0.091262 → 0.044460 (-51.3%) | 0.099485 → 0.039953 (-59.8%) |
| Original recordings: whole-track lip-gap RMS | 0.358713 → 0.356957 | 0.302649 → 0.294547 |
| Paired recordings: pause lip-gap RMS, pilot off | 0.795755 → 0.607237 (-23.7%) | 0.553474 → 0.431910 (-22.0%) |
| Paired recordings: pause lip-gap RMS, pilot on | 0.763650 → 0.606085 (-20.6%) | 0.487443 → 0.392190 (-19.5%) |

End-transition coefficient acceleration RMS falls by 74.6% / 67.6% on the original fresh/test recordings. Results use each full quiet interval and a 50 ms border when measuring transition steps; paired transition summaries include both internal pauses and final tails. The numerical reference remains full native lip geometry, not an ARKit reconstruction.

Every protected speech-contact frame and speech-run contact mask is unchanged in all six evaluation groups, with the pilot both off and on. Each P/B/M advance-window classification is identical to the previous version (including exclusions): original fresh TP/FP/FN stays 18/2/11; original test stays 10/1/8. No complete-track contact count or isolated quiet-contact count increases. Coverage and selected match counts are unchanged. Accepted recorded-pause frames are bit-identical with the smoother fallback enabled. The envelope was developed on the seven development recordings and three development pairs; the fixed fresh/test suites provide comparisons and regression checks.

Rendered reviews cover the previously unsupported P→pause in “Bring the cup. Then shut the door.” and the ending of “The little dog slept under a blanket.” They show the slower settling clearly, while also showing that synthetic relaxation does not reproduce every feature of the native motion. Live browser generation and playback were checked with the new default enabled and the recorded-pause pilot unchecked.

## Additional phoneme substitutions: opt-in pilot

**S/Z, SH/ZH and K/G fallbacks (pilot)** now enables only those three recommended pairs alongside the existing P↔B pair. The browser checkbox is checked by default and can be turned off for comparison. F/V, TH/DH and T/D are retained in the CLI/API for experiments but are not enabled by the browser checkbox. Every pair is symmetric, can replace the first, middle or final phoneme, and permits only one substitution per triple. Exact triples still take priority at the lookup position; approximate triples compete with exact doubles using the existing timing/join cost and substitution penalty. This changes animation lookup only, not generated audio or TinyAligner's phonemes and timings. M, vowels and other consonant groupings remain excluded. The full-six-pair results below document the earlier trial; the three-pair comparison at the end explains the browser recommendation.

API callers can import `RECOMMENDED_VISEME_PAIRS` from `src/lib/dictionary/viseme-pairs.mjs` to use the browser's subset. For experiments, pass `visemePairs: ['P/B','F/V']` to isolate a pair or import `ALL_VISEME_PAIRS` for the complete set. An empty list disables substitutions; unknown pair identifiers are rejected. Selected approximate triples are counted separately from exact triples and doubles, with per-pair counts in the UI and comparison reports. The `planAnimation` API default remains P/B only; the browser explicitly requests the recommended subset through its checked checkbox. This option is independent of the recorded-pause pilot.

The seven development and fixed 24 fresh / 12 test sentences were compared against current P/B playback with advance guidance, the opening fix and 200 ms neutral transitions enabled in both variants. Each additional pair was tested separately and then all together. The development set selected no new substitutions. Only S/Z and K/G were selected on the fixed fresh/test sets, so those sets cannot assess the other four pairs.

Twelve additional sentences in `scripts/extended-viseme-prompts.mjs` exercise the remaining opportunities. These prompts were selected using the pronunciation lexicon and missing dictionary triples, not native animation scores. They are a targeted challenge set, not a replacement for the fixed test set or an unbiased estimate of general quality. They do not overlap training or existing evaluation transcripts. Phonon generates their speech, full TinyAligner aligns it, and full native Audio2Face supplies new references; these recordings are not added to the dictionary.

```sh
node scripts/compare-fallback.mjs --extended-visemes --output .local/extended-fresh.json
node scripts/compare-fallback.mjs --extended-visemes --test-set --output .local/extended-test.json
node scripts/compare-fallback.mjs --extended-visemes --substitution-pairs F/V --development-set --output .local/fv-development.json
node scripts/prepare-viseme-challenge.mjs
"$A2F_PYTHON" scripts/label.py .local/viseme-challenge --runner "$A2F_RUNNER" --model "$A2F_MODEL" --rig "$A2F_RIG" --geometry-reference
node scripts/compare-fallback.mjs --extended-visemes --challenge-set --output .local/extended-challenge.json
```

`--substitution-pairs` accepts comma-separated canonical pair identifiers and automatically retains P/B. Omitting it tests all seven pairs. `--save-tracks` writes baseline, expanded, native ARKit-fit and audio review tracks under `.local/extended-visemes/review/`. Exact, approximate and single fallback interval counts remain separate; mixed exact/approximate overlaps are classified as approximate. No missing intervals were recovered on these sets: their single fallback/miss counts stay unchanged. Selected exact-triple counts are unchanged, while some doubles are replaced or overlapped by approximate triples.

| All additional pairs together | Fresh 24 | Test 12 | Challenge 12 |
| --- | ---: | ---: | ---: |
| Lip-gap RMS change | -0.70% | +0.14% | -0.17% |
| Join step-error RMS change | -2.05% | -0.38% | -2.99% |
| Additional approximate triples selected | 6 | 1 | 14 |
| Matched / extra / missed native contact episodes, before and after | 46 / 26 / 16 | 36 / 10 / 10 | 27 / 12 / 2 |

| Isolated pair, measured over the same 12 challenge sentences | Selected matches | Lip-gap RMS change | Join error change |
| --- | ---: | ---: | ---: |
| F/V | 2 | +0.19% | +0.25% |
| S/Z | 2 | -0.06% | -0.17% |
| TH/DH | 1 | +0.51% | -0.23% |
| SH/ZH | 2 | -0.28% | -2.06% |
| T/D | 3 | +0.16% | +0.50% |
| K/G | 4 | -0.69% | -1.28% |

Lower error is better. Full-combination effects need not equal the sum of isolated effects. S/Z alone slightly worsens fixed test-set shape error; F/V, TH/DH and T/D worsen challenge shape error. For example, the expanded version increases whole-sentence lip-gap error by 4.1% on “Wafers fell from the silver tray.” and 7.8% on “Ethan thought the path was lethal.” Sharing a broad viseme does not establish interchangeable motion through the surrounding sounds. No pair-specific penalty was tuned on these reference outputs.

Across all 28 split/pair comparisons, the per-sentence event audit retains every previously matched native reference closure and pairs every predicted extra episode to an existing baseline extra episode. No extra-contact count increases. The combined variants' advance-window TP/FP/FN and exclusion counts are also unchanged. This is a measured regression check, not a runtime guarantee for untested speech. Quality metrics use nine vertical lip gaps and join motion; they do not fully score tooth/tongue visibility or lip protrusion. Functional tests cover both substitution directions and every triple position, exact-triple priority, single-substitution limits, disabled defaults, invalid configuration and silence boundaries. Native quality examples cover every pair, but not every direction/position combination.

Side-by-side rendered reviews include “Please leave the keys beside the sink.” (a substantial measured improvement with S/Z and K/G selected) and “Wafers fell from the silver tray.” (an F/V regression). The full native reference is rendered through its ARKit fit. Browser generation with the additional-pair checkbox enabled was verified, including the per-pair match counts. The browser now exposes the narrower recommended subset; the full six-pair trial remains reproducible through the CLI/API.

Comparing just S/Z + SH/ZH + K/G with all six additional pairs gives identical output on the fixed fresh/test sentences because F/V, TH/DH and T/D are not selected there. On the challenge set, the three-pair subset reduces lip-gap error by 1.03% and join error by 3.52%, versus 0.17% and 2.99% for all six. Both retain P/B, and all closure-event audits pass. For a fair cross-configuration join comparison, add `--join-reference-pairs all` to the subset run and `--join-reference-pairs S/Z,SH/ZH,K/G` to the full run. That measures both over the same union of baseline/subset/full join anchors; otherwise each two-way comparison can use slightly different join windows.

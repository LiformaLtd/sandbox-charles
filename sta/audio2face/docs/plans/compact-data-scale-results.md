# Compact student: memorization and real data scale

The unchanged compact network benefits from a substantially larger corpus and
training budget. Across three fixed optimizer seeds, expanded-data long training
reduced original-level macro-voice lip-gap RMS by **7.22% versus published v4**
and **5.31% versus the matched original-data long control**. Both engines improved.
The browser model remains **2,251,827 bytes / 331,047 parameters**.

This is a useful research result, not an automatic model promotion. The original
seed-37 pilot missed its predeclared 5% control-relative screen (4.45%). We kept
that failure and registered a separate, post-pilot stability extension using the
two already-listed seeds, without changing the recipe. The three-seed mean is
exploratory evidence on the same inspected cohort, not a new independent test.
Published v4 and its CDN assets are unchanged. No seed was chosen for release
using confirmation scores.

## First establish learnability

Four existing training recordings cover Phonon Easton /m/, Phonon Zoey /b/,
Supertonic M1 /p/ and Supertonic F1 vowels: 27 windows / 683 frames (~11.38 s).
Both diagnostic arms start from published v4 and train 1,500 full-batch AdamW
updates. These are training-set fits, deliberately not held-out evidence.

| Training-set result | Student → fitted target RMS mm | Student → raw teacher RMS mm | Contact F1 |
|---|---:|---:|---:|
| Initial v4 | 2.14895 | 2.09544 | 0.80938 |
| Existing mixed loss, selected step 200 | 0.37449 | 0.42152 | 0.98328 |
| Pure fitted-coefficient MSE, selected step 1500 | 0.17525 | 0.41645 | 0.95406 |

The fitted target itself differs from raw geometry by 0.37221 mm. Both arms pass
the diagnostic limits (≤0.5 mm to fit, ≤1 mm to teacher). The architecture can
represent much better trajectories on these examples; this does not prove enough
capacity for arbitrary voices. Lower fitted-target RMS does not necessarily give
better closure F1. Neither diagnostic export is deployable.

## What changed

[The frozen grid](compact-data-scale-grid.md) controls initialization, architecture,
precision, normalization, losses, sampler proportions and selection policy.

- Primary originals increase 256 → 3,328 (13×); gain-conditioned primary examples
  increase 768 → 3,840 (5×). The 549 clean historical replay clips are unchanged;
  the 13× figure does not apply to the total corpus including replay.
- 3,072 new training recordings use eight existing training voice presets,
  balanced across Phonon/Supertonic and catalogue genders. New training speech
  totals about 2.79 hours. Each recording gets one balanced original/half/boost
  level and its own teacher output; changed waveforms never inherit labels.
- There are 192 new development recordings (576 gain-labelled cases), combined
  with 288 old development cases and the separate Easton regression guard.
- Confirmation has 88 disjoint texts × four known training-held-out voice presets
  = 352 new recordings / 1,408 original, half, boost and quarter cases. These are
  new transcripts and recordings, not newly discovered voice presets.
- 512 training texts come from CMU ARCTIC; 256 are authored across articulation
  and dialogue families. Dev/test use separate 32/64 ARCTIC and 16/24 authored
  texts. Authored variations share templates. Thus this changes volume *and*
  linguistic distribution, rather than isolating count alone.
- Audit excluded 1,549 retained prior normalized texts across 42 manifests.
  All 3,616 new recordings and 5,056 new labelled examples completed source,
  transcript, waveform, teacher-recipe and target checks.

Unchanged ARCTIC sentences came from a pinned public text mirror, commit
`c51cb971036a127359b43c621f26ca5081eab13e`; the source bytes, hash and CMU license
notice are retained with the local artifacts. No original human recordings were
used. See the [original CMU ARCTIC report](https://festvox.org/cmu_arctic/cmu_arctic_report.pdf).

Each data trajectory runs 3,720 updates. Short-budget selection uses only its
exact first 744 updates; both budgets share the same long cosine schedule.
This short schedule differs from earlier studies with a complete 744-step decay.
Each update draws 48 voice-balanced primary and 16 clean replay windows.
Selection uses development data only, initially and every 124 steps. All four
exports were locked before fresh scoring. Confirmation data never selects a
checkpoint, learning rate, source mix or threshold.

## Matched grid: original pilot seed 37

| Model | Original-level RMS mm | Contact F1 |
|---|---:|---:|
| Published v4 | 2.91451 | 0.77447 |
| Original data, short budget | 2.91481 | 0.77233 |
| Original data, long budget | 2.85131 | 0.77530 |
| Expanded data, short budget | 2.80493 | 0.77904 |
| Expanded data, long budget | 2.72455 | 0.78855 |

Data reduces RMS 3.77% at short budget and 4.45% at long budget. More updates
reduce it 2.18% with original data and 2.87% with expanded data. The expanded
long model improves 6.52% over deployed v4. Both engines, half/boost mean,
contact and historical regression guards pass; only the 5% primary screen fails.
Keep that pilot result intact.

## Fixed-seed stability extension

| Seed | Original-data long RMS mm | Expanded-data long RMS mm | Gain vs control | Gain vs v4 |
|---|---:|---:|---:|---:|
| 37 | 2.85131 | 2.72455 | 4.45% | 6.52% |
| 73 | 2.86645 | 2.70969 | 5.47% | 7.03% |
| 101 | 2.84964 | 2.67834 | 6.01% | 8.10% |
| Mean | 2.85580 | 2.70419 | **5.31%** | **7.22%** |

Mean contact F1 rises 0.77598 → 0.78881 (v4: 0.77447). Well-timed /m b p/
closures average 395.3 → 436.3, **10.37% more than the matched control**. Published
v4 has 419 hits, so the improvement versus v4 is **4.14%**, not 10.37%. Each
candidate improves each of /m/, /b/ and /p/ versus v4. Counts use the existing
anticipatory closure audit, not a new timing threshold fitted to these results.

Mean Phonon RMS improves 6.49% versus control / 8.92% versus v4. Supertonic
improves 4.22% / 5.63%. Both authored and ARCTIC subsets improve. Opening-only
RMS also improves: v4 3.3040 mm; control seeds 3.1850/3.1895/3.1779; expanded
3.0611/3.0568/2.9975. This is not simply more closed frames winning a score.

A paired crossed bootstrap (2,000 draws, fixed seed 20261107), resampling
optimizer seeds, transcripts and voice presets within engine, gives a 3.46–6.86%
interval for control-relative RMS reduction, and +0.00780–0.01842 for contact F1.
These are exploratory intervals: only four known confirmation presets and the
same inspected population. They do not establish generalization to every voice.

## Backend and browser checks

Torch 2.4.1 MPS preflight exposed stride-dependent geometry forward errors and
inconsistent gradients/optimization even after making geometry contiguous. No
matched-grid trajectory used that backend. Native MLX 0.31.2 training was checked
against CPU, including objective, gradients and first AdamW update. Disable
`MLX_ENABLE_TF32` before import: default TF32 materially widened the comparison.
Final float32 forward max delta was 3.58e-7, gradient relative L2 1.03e-6 and
first-update parameter max delta 2.00e-6. Exports use the unchanged CPU PyTorch
waveform model, not a new browser runtime. Memory-mapped caches omit unused large
speech features; the reference-cache golden comparison had zero differences.

All graphs retain identical operators, file size, **25 explicit GPU buffers /
1,492,236 bytes**, and two reusable pipelines. Twelve full-track native/browser
fixtures cover both engines, gains, female voices and the complete packed phrase.
Three settled WebKit runs plus first-clip repeats cover all five seed-37 models;
extra seed-73/101 candidates also pass. Maximum coefficient error is 8.65e-7,
with no nonfinite or unexpected zeroed frames. Repeat generation is deterministic.

Warm 500 ms emitted blocks had median-of-six packed-track medians of 1.28 ms
for v4, 1.35 ms for the long control and 1.34 ms for expanded-long on this M5 Max.
Observed ranges overlap substantially (v4 1.06–2.22 ms, control 1.04–1.46 ms,
expanded 1.14–1.48 ms). No meaningful speed change is established. Earlier
unsettled runs are preserved separately; rig loading must finish before timing.

Sampled desktop WebKit WebContent+GPU physical-footprint peaks were 526.5 MiB
(v4), 520.3 MiB (control) and 534.4 MiB (expanded). These are simultaneous process
sums, sampled every 20 ms during bounded repeated prerecorded inference with
James continuing to animate at idle. TTS is excluded, as are network/browser
processes. Sampling can miss brief spikes; this is not an iPhone or full-pipeline
under-2-GB guarantee. The variation does not establish a memory saving/regression.

Sixteen synchronized four-lane review cases compare the fitted teacher, v4,
control and expanded model. Audio and exact seeks passed browser verification.
This functional review is not a blinded perceptual quality study.

## Artifacts and next decision

All generated data/models are ignored under `.local/compact-scale-v1`, including
memorization, corpus provenance, training/selection locks, frozen sources,
checkpoints/optimizer/RNG states, complete scores, browser parity/timing/memory
and `review.html`. Repeat outputs are in `repeats/s73` and `repeats/s101`.

Baseline SHA-256: `5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd`.
Training-protocol SHA-256: `ffb2c993113ec64bf75eda2825fdbac5a6bbd7a40520960277683cc7b9ac3c3f`.
Seed-37 expanded-long: `407739196cd98c89ec619b6af527f1021a3702cf8b1213c006f86408d1701137`.

See [reproduction instructions](../../scripts/research/data_scale/README.md).
The research branch also retains the previously deployed Pocket/stale-chunk
fixes so a later release cannot accidentally revert them. The next release
needs a documented selection rule and listening/visual checks; do not choose
seed 101 simply because it has the lowest confirmation error. Keep current
production v4 until a deliberate model publication step.

## Requested scale-r3 release

Following the research report, the user requested updating v4 and publishing it
for iPhone testing. Retain seed **37**, the original pilot identity, with its
**development-selected step 2976**; do not select seed 101 from confirmation.
This individual export improves original-level RMS **6.52% versus coverage-r2**,
not the three-seed mean of 7.22%. Its control-relative pilot screen remains failed.

Immutable CDN prefix: `models/speech-to-animation/lf-sta/arkit-student-v4-scale-r3`.
`publish_v4.mjs` pins the selected hash and step, publishes the model, existing
license/notice files and checksum manifest, and downloads each to verify bytes.
The old coverage-r2 URL remains available. The hosted v4 redirect now uses
scale-r3; the UI retains v4 and WebGPU as defaults, with idle animation unchanged.
Physical iPhone testing is the purpose of this release, not a completed claim.

Deployment `dpl_4QxPSr3ZTowV1oWPXwjx7B9Lv8bU` was built from release commit
`2876a76`, staged as production without domains, then promoted after desktop
WebKit checks passed for Phonon, Supertonic and Pocket speech/audio playback.
The tested build is live at https://james-audio2face.vercel.app/ on Liforma.
Checks verified the exact model SHA and size through the hosted asset redirect,
v4/WebGPU/Phonon defaults, animated idle canvas changes and no page errors.
Local v4 weights also match scale-r3. Prior immutable research baselines remain
unchanged. Browser fixtures emulate an iPhone user agent; physical iPhone memory
and repeated-generation stability still require on-device testing.

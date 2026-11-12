# Articulation-state auxiliary supervision: results

The training-only state head slightly reduced lip-gap RMS but did not improve
closure reliability. **Keep published v4 coverage-r2.** This one-seed pilot failed
the frozen quality screen and is not a release candidate.

## Intervention and integrity

Both arms started from published v4 coverage-r2 and retained its 96-channel
student, 64-bin log-mel frontend, context, training/replay data, normalization,
base losses, seed 37, learning rate and 24 × 31 update budget. The state arm added
a training-only linear head (96 → 4), with weighted cross-entropy coefficient
0.1. Teacher geometry supplied closing/closed/opening/open labels; phoneme
alignment did not define transition times. Both arms instantiated the same
modules; only the auxiliary loss differed. The auxiliary head was removed at
export, not loaded and disabled inside the browser.

Both selected epoch 18 (558 updates) before fresh test scoring. The frozen-source,
data/replay/sampler and initial-development audits passed. The control reproduced
the previous capacity study's 96-channel export **byte for byte**, providing an
additional baseline check. Auxiliary loss gradients reached the encoder (norm
1.891) and head (2.140) on a 16-window diagnostic batch. Final auxiliary head
weights differed from the untouched control head by norm 0.3986.

Primary valid training state counts were 22,914 open/steady, 40,971 closing,
38,055 opening and 33,651 closed/contact. These are fixed engineering labels,
not reviewed physiological annotations. Frames lacking either valid neighbour
were ignored, preventing padding and block boundaries from creating transitions.

## Fresh confirmation

24 new transcripts × four known training-held-out presets produced 96 recordings,
48 per engine. Teacher-labelled original/half/headroom-boost/quarter levels gave
384 cases. These were new texts/audio, not unseen voices. Exact normalized text
overlap with prior known corpora, including capacity confirmation, was excluded.
Checkpoints were selected before this cohort was scored. RMS is macro-voice
James lip-gap error in millimetres across nine vertical gap measurements.

| Audio condition | Published v4 | Matched control | State supervision |
|---|---:|---:|---:|
| original | 2.6594 | 2.6317 | 2.6219 |
| half | 2.8513 | 2.7942 | 2.7811 |
| boost | 2.6875 | 2.6664 | 2.6604 |
| quarter | 3.3071 | 3.2557 | 3.2445 |

Original RMS improved **0.37%** versus control, an absolute reduction of about
**0.010 mm**: Phonon improved 0.44%, Supertonic 0.32%. Half/boost mean improved
0.35%. This is far below the frozen 5% screen, with no independent-seed evidence
of a useful perceptual gain. Comparing only against published initialization
would incorrectly credit extra training to the auxiliary task.

| Original-level measure | Published v4 | Matched control | State supervision |
|---|---:|---:|---:|
| Contact F1 | 0.75133 | 0.75151 | 0.74948 |
| Opening-only RMS (mm) | 3.0337 | 2.9666 | 2.9533 |

The contact-F1 guard failed. Both engines' F1 declined despite their lower RMS.
Closure timing uses actual teacher contact, with phonemes only associating the
event in a 100 ms lead / 35 ms trail window. Timing MAE is conditional on overlap,
so it must be interpreted alongside hits. All arms share the same associated
teacher events: 73 /m/, 114 /b/ and 120 /p/.

| Phone | Control / state overlap hits | Control / state well-timed hits | Onset MAE ms, control / state | Release MAE ms, control / state |
|---|---:|---:|---:|---:|
| /m/ | 54 / 55 | 42 / 41 | 19.44 / 20.00 | 14.51 / 14.85 |
| /b/ | 92 / 91 | 81 / 80 | 13.59 / 13.37 | 15.04 / 14.65 |
| /p/ | 94 / 94 | 81 / 80 | 18.09 / 18.26 | 11.35 / 11.35 |

No phoneme category gained well-timed hits. Small opening/timing changes did not
translate into more reliable complete closures.

## Previously inspected regressions

These are regression sets, not fresh evidence. Original-level macro RMS:

| Cohort | Published v4 | Matched control | State supervision |
|---|---:|---:|---:|
| historical | 2.7546 | 2.7278 | 2.7274 |
| coverage_regression | 2.6031 | 2.5700 | 2.5613 |
| capacity_regression | 2.6024 | 2.5524 | 2.5471 |

Historical/coverage RMS and F1 guards passed. The small gains on previously
inspected material did not compensate for the fresh contact regression.

## Browser and visual checks

Both exports retained **331,047 parameters, 2,251,827 bytes and identical ONNX
runtime operators**. The production WebGPU runner/shaders were unchanged. Three
fresh desktop WebKit runs compared each model with native inference across
eight fresh engine/gain cases and both full “packed” tracks. Maximum absolute
coefficient difference was **7.45e-7**; no nonfinite, bad or unexpectedly zeroed
frames occurred.

| Browser cost | Matched control | State supervision |
|---|---:|---:|
| Explicit model GPU buffers | 25 | 25 |
| Explicit model GPU bytes | 1,492,236 | 1,492,236 |
| Warm long-track block median | 1.01 ms | 1.03 ms |
| Range of six per-track medians | 0.98–1.06 ms | 1.02–1.06 ms |

Timing uses the local M5 Max, after training finished, excluding the first block
per track. The tiny timing difference is not an established speed regression:
operators are identical. A typical block emits 500 ms; these timings exclude
TTS, rendering, startup and audio scheduling. Buffer counts are not whole-page
peak memory, driver allocations or a physical-iPhone measurement.

Sixteen comparison cases were fixed before scoring: /m b p/ from all four
presets, plus half-level /p/ and quarter-level vowel speech from both engines.
The synchronized four-lane James review includes fitted full teacher, published
v4, matched control and state arm, with raw teacher-gap plots. Browser checks
verified audio, all cases, exact seek frames and no page errors. Inspected
mouth screenshots showed similar student poses, not an obvious closure win;
this is not a blinded perceptual study. Normal demo idle animation is unchanged.
Review URL on the existing local server:
`http://127.0.0.1:5178/.local/compact-state-v1/review.html`.

## Post-hoc diagnosis and next hypothesis

A separate error decomposition on the new original-level recordings found
macro-voice fitting RMS **0.374 mm**, student-to-fitted-target RMS **2.627 mm**,
and student-to-raw-teacher RMS **2.622 mm**. Thus the remaining gap on this cohort
primarily lies in predicting the motion, rather than projecting teacher geometry
onto the available James controls. Fitting is not a strict error floor: the
student also learns raw geometry losses, and cross-error terms can be negative.
This diagnostic was not used to select a checkpoint or change the quality gate.

Extra width and a training-only state classifier both made marginal changes.
The next isolated hypothesis should change how motion history is represented:
a tiny stateful temporal decoder or short learned motion prior, while preserving
the lightweight audio frontend. That is an untested hypothesis, not an assumed
winner. Keep any new state small, measure reset/streaming correctness, and require
fresh closure/per-engine gains before changing deployed v4. Do not tune these
state thresholds or weight against this now-inspected confirmation cohort.

## Reproduction and artifacts

[Protocol](compact-articulation-state.md) ·
[Commands](../../scripts/research/articulation_state/README.md)

Ignored root: `.local/compact-state-v1`. Retain protocol/source/data locks,
complete optimizer checkpoints, `head-gradient-check.json`, `training-audit.json`,
`selection.json`, `evaluation-*.json`, `summary.json`, `fitting-diagnostic.json`,
`browser/parity/run-*.json`, `review-verification.json`, and review tracks/screenshots.

- Published v4: `5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd`
- Matched control: `96535c26d4f3372584cceeb46f023e541b0638107c51c45b99dab71fbba9034b`
- State arm: `2468d5d3326a7f3dac9d3941bbc2750c33f2fa8cc0dcb7ea667fb45298adf494`

No experimental weights were deployed or published to the CDN in this phase.

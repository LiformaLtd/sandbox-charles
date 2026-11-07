# Compact v4 candidate validation results — 2026-11-07

Protocol: [compact-v4-validation.md](compact-v4-validation.md).
Earlier exploratory results: [multi-voice pilot](compact-multivoice-results.md).

## Outcome

The unchanged multi-voice + clean-replay recipe passed the predeclared desktop
engineering screen across three fine-tuning seeds and 96 fresh clips from four
new voices. The fresh confirmation gain is **5.4% lower lip-gap RMS**, smaller
than the exploratory pilot's roughly 11%. Contact F1 also improved. This supports
a v4 candidate; it does not establish universal voice coverage, full-teacher
quality or physical-iPhone reliability.

The representative is **seed 73, epoch 19**, selected by the lowest five-voice
development score before fresh test evaluation. Seed 101 had the best fresh RMS
but was not substituted. All exports retain 331,047 learned parameters and
2,251,827 bytes, with no runtime aligner or additional speech encoder.

Candidate SHA-256:
`b4a275acf78b982fb6ec6fc55333c5cf25613e72b6d9b3d80e95d6e74ac6261d`.
The frozen local export is `.local/compact-v4-validation/v4-candidate.onnx`.
The active demo models were not replaced or deployed.

## Confirmation and independent fine-tuning seeds

Trevor/Pippa from Phonon and M3/F3 from Supertonic 3 each spoke the same 24 new
texts. They were absent from known compact training and the pilot. All normalized
transcripts were checked against historical compact/pilot/confirmation manifests.
Shared TTS revisions, resampler, aligner, fitter and architecture sources matched
the pilot. The teacher instrumentation parity check passed again. Teacher labels
retained unshifted geometry timestamps and the corrected 100 ms event association.

Seeds 37, 73 and 101 used identical v3 initialization, data, 744 optimizer updates,
losses and development-based selection. These replicate fine-tuning randomness,
not the entire original pretraining history. The protocol/definition/model hashes
were frozen before the first confirmation evaluation. No parameter tuning or
combination selection used the fresh set.

Scores below average voices equally. Lip-gap RMS is against raw teacher geometry
at nine lip pairs, converted from source centimetres to millimetres.

| Model/run | Selected epoch | Fresh lip-gap RMS (mm) ↓ | Fresh contact F1 ↑ |
| --- | ---: | ---: | ---: |
| Released v3 | — | 3.042 | 0.7604 |
| Candidate seed 37 | 17 | 2.897 | 0.7816 |
| Candidate seed 73, representative | 19 | 2.880 | 0.7836 |
| Candidate seed 101 | 24 | 2.859 | 0.7833 |
| Candidate mean | — | **2.879** | **0.7828** |

Across-seed RMS standard deviation was 0.019 mm. A paired crossed bootstrap
(4,000 draws, resampling seed, text and voices within engine) gave a candidate
minus v3 RMS interval of **[-0.301, -0.025] mm**, and F1 difference
**[+0.0082, +0.0376]**. These intervals reflect this small cohort, with only
three seeds and two new voices per engine; they are not population guarantees.

| Engine, mean across seeds | v3 RMS (mm) | Candidate RMS (mm) | Relative change |
| --- | ---: | ---: | ---: |
| Phonon | 3.049 | 2.810 | -7.8% |
| Supertonic 3 | 3.034 | 2.947 | -2.9% |

Per-voice RMS changes were Pippa -10.0%, Trevor -5.8%, M3 -6.1%, F3 +0.1%.
F3 is effectively tied on this metric, rather than evidence that every voice
improved. All four voices improved framewise contact F1.

## Historical regression and engineering checks

On the old 48-clip Easton set, v3 RMS was 2.580 mm. Candidate mean was 2.589 mm,
a 0.34% increase; mean F1 improved from 0.7618 to 0.7706. The representative's
Easton RMS was 2.583 mm and F1 0.7720. Those are historical regression checks,
not an independent fresh test.

All frozen point-estimate checks passed: fresh macro RMS reduction at least 5%;
no engine-average RMS regression; no fresh macro contact-F1 regression; no new
voice RMS regression over 2%; historical Easton mean RMS regression at most 2%
and F1 delta at least -0.01. Passing this screen supports on-device review,
not an automatic release or a stronger causal claim.

## Fresh closure events

The corrected policy separates association from prediction timing. Teacher
episodes are associated with phones using 100 ms pre-padding/35 ms post-padding;
shared, multiple and clip-truncated associations are excluded and reported.
Predictions are matched one-to-one against actual teacher contact times. No
global 50 ms time shift is applied. The denominator includes unique teacher
episodes lasting at least two 60fps frames, not every acoustic labial.

| Sound | Teacher episodes | v3 overlap hits | Representative overlap hits | v3 / representative well-timed hits |
| --- | ---: | ---: | ---: | ---: |
| `/b/` | 113 | 74 | 89 | 67 / 80 |
| `/p/` | 98 | 52 | 78 | 44 / 62 |
| `/m/` | 92 | 50 | 65 | 40 / 54 |

Overlap alone is not timing accuracy. "Well timed" requires temporal IoU at least
0.5 and onset error at most two frames (33.3 ms) relative to the teacher. Signed
onset/release errors and 45/150 ms association sensitivity remain in the raw
evaluation reports. These are operational teacher-agreement criteria, not a
biological correctness test.

## Three fresh generations on one browser page

Desktop M5 Max / macOS WebKit 2388 with iPhone 13 emulation. Each model/engine
combination used a fresh browser process, prepared TTS and STA once, then
generated three **different** texts on that same page. This bypassed the demo's
identical-text replay shortcut. No training ran during measurements.

Every sequence verified new non-silent audio, fresh STA inference, rendered frame
progression, one navigation, no page errors, and continuing idle poses after each
play. Model GPU allocations stayed exactly **1,492,236 bytes, 25 buffers and two
compute pipelines** throughout; no model teardown or forced garbage collection
was used between plays. Candidate CPU/WebGPU parity differed by at most 2.69e-7
across 9,360 coefficients.

| Model / TTS | Play 1 peak (MB) | Play 2 peak (MB) | Play 3 peak (MB) |
| --- | ---: | ---: | ---: |
| v3 / Phonon | 905.8 | 860.5 | 818.0 |
| Candidate / Phonon | 883.4 | 857.9 | 813.0 |
| v3 / Supertonic 3 | 1584.0 | 1552.7 | 1522.3 |
| Candidate / Supertonic 3 | 1575.4 | 1547.1 | 1518.7 |

The peaks decreased over these three generations; cumulative growth was not
observed in this short check. It is not a long-duration soak test. Individual
per-play median warm animation-block times ranged 1.92–4.46 ms, for 30 emitted
frames / 0.5 s. The candidate has identical inference work to v3; these noisy
single-sequence measurements do not establish a model speedup or memory saving.

Sampling begins after the initial rig is ready and covers model preparation and
all three plays. Peak is the sum of sampled WebContent/GPU physical footprints
at 20 ms intervals, not unique total machine memory or a guaranteed browser
limit. TTS startup/generation is excluded from warm block time. Local pinned
Supertonic assets replace Vercel redirects in the isolated Vite harness. The
result does not certify cold rig-loading peaks, other browsers or a physical
iPhone. Supertonic still costs substantially more full-page memory than Phonon.

## Visual review and remaining work

Six protocol-defined fresh cases provide synchronized teacher-fitted/v3/selected
candidate lanes with the same speech clock. Cases cover early contact, vowel
opening, fricatives and conversation. A stationary-head inspection mode does
not change the normal demo's idle animation. This review is not a blinded human
preference study, and the compact model still misses some teacher opening detail.

Run the isolated harness and open `/review.html` for the comparison. The script
checks playback, seeking and all six clips; local screenshots/traces and the
verification JSON are retained with the artifacts.

The next release gate is **physical iPhone testing**, including repeated distinct
generations with WebGPU and idle animation. Neither desktop emulation nor a
sub-2-GB desktop process sum substitutes for that. Further model work should
expand acoustic/lexical variation and test properly teacher-labeled gain
variation; the fresh confirmation set must not become a tuning set.

Artifacts remain outside Git under `.local/compact-v4-confirmation` and
`.local/compact-v4-validation`. Shared training runs remain under
`.local/compact-multivoice/runs/control-replay-s{37,73,101}`. See the research
README for the freeze/select/evaluate/browser/review commands.

## Demo deployment — 2026-11-07

At the user's request, the representative was published to the versioned
`arkit-student-v4-candidate` CDN prefix with matching checksum, manifest and
upstream notices. It is now selectable alongside v3 in **Liforma /
james-audio2face** at
[the v4 demo](https://james-audio2face.vercel.app/?model=compact-v4&sta=webgpu).
WebGPU remains the default runtime. Switching releases the prior model and
retains dialogue/settings; idle motion continues.

The tested production deployment is `dpl_DEbfBfYreaaVaERGRByxviKgsoiV`.
Hosted iPhone-emulated WebKit verified v4 → v3 → v4 playback, retained dialogue,
the correct CDN URLs, the dedicated two-pipeline runner, audio and idle motion.
It also verified Supertonic playback after correcting its hosted WebKit redirect
failure: its worker now resolves pinned upstream model/style URLs directly,
using the same resolver as Phonon and STA. The staged build was tested before
promotion and the main alias was tested after promotion. Physical iPhone
validation is still pending; deployment is not evidence of an on-device test.

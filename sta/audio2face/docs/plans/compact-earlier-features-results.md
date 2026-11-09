# Earlier learned features: no useful residual-learning gain

See the [design](compact-earlier-features.md) and
[reproduction guide](../../scripts/research/earlier_features/README.md).
Evidence stays ignored under `.local/compact-early-v1`. No model is promoted,
published or deployed. Keep the original v4, WebGPU default and animated idle.

## Isolated comparison

The early arm uses96-channel features directly after v4's first5-tap audio
convolution/GELU, before its eight residual temporal blocks. The late control
uses the original final96-channel features. Both retain exactly the same frozen
39-channel base prediction, same135→96→39 pointwise residual head,16,839 trainable
parameters, zero-output initialization, loss, training budget and selection rule.
Only the source features and their training-only RMS normalization differ.
The original base still supplies its full context/lookahead. This does not test
raw log-mel correction or a new temporal decoder.

One common float32 base-prediction cache is shared by both arms. Early activations
are newly cached; late activations are hash-checked against the preceding pilot.
All original data-array, new cache and model hashes are frozen before training.
Both initial waveform exports match v4 exactly on the random preflight waveform,
with identical initial objective. Both arms run7,440 updates and select step1240.
Their scheduled sampler hashes match
`589ce3223580562376a65c7a6dc5876ae0359524f120d90478e167b9f83f44f3`.
The late arm's selected head arrays reproduce the preceding residual pilot exactly
(max absolute delta0), validating the migrated cache/input path.

The exporter materializes PyTorch's aliased identical kind flag before training
so graph/initializer contracts match. A training-fit adapter initially omitted
the separately cached base predictions; its pre-score error and old source lock
are archived under `evaluation-preflight-repair`. It was corrected and relocked
before any fresh score, without changing either model or training source.

## Fresh-speech quality

The24 new transcripts (12 /m,b,p/vowel-targeted,12 Arctic) exclude2,797 previously
planned/observed texts. Four voice presets outside training span Phonon and
Supertonic. The96 original recordings have384 independently teacher-labelled gain
cases. Corpus completion and model selections are locked before scoring. Old
historical/feature cohorts are regression checks, not fresh evidence.

| Model | Original training gap RMS | Fresh gap RMS | Contact F1 | /m,b,p/ hits |
|---|---:|---:|---:|---|
| Published v4 | 1.83882mm | 2.46963mm | .80884 | 65 / 56 / 56 |
| Late-feature control | 1.81740mm | 2.46226mm | .80880 | 69 / 57 / 56 |
| Early-feature head | 1.81850mm | 2.46190mm | .80791 | 69 / 57 / 56 |

The early head improves fresh gap RMS only.01452% over late, far below the3%
predeclared gate and not an established practical gain. Training RMS is slightly
worse. Phone-event hits tie182 for the two heads versus177 for v4, but framewise
contact F1 worsens, showing why event hits alone do not establish preserved
closures. Overall gain over published v4 is only.313%, versus.298% for late.

Phonon gap/F1: late2.39448/.81855, early2.39394/.81862.
Supertonic: late2.53005/.79904, early2.52986/.79721. Both tiny gap changes are
negligible; Supertonic contact F1 worsens. False-closure rate increases
.066774→.067239 and missed-contact rate.179598→.180183. Conditional onset MAE
worsens17.133→17.451ms, while release MAE11.686→11.774ms remains within the
predeclared.1ms allowance. Timing is relative to actual teacher closures, with
up to100ms anticipatory association; compare it alongside event counts.

RMS-gain, macro/per-engine F1, false/missed-contact and onset guards fail. Phone
hits, per-engine gap non-regression, release timing and historical/feature guards
pass. This is one seed and one fresh cohort; no test-driven sweep follows. Do
not compare the cohort's absolute scores directly with earlier cohorts.

## Export and runtime integrity

Both models contain347,886 parameters and2,322,775 bytes, below the frozen2.35MB
budget. Every original base tensor, frontend/interpolation value and channel map
matches v4 exactly. Both exported graphs match after serialization normalization.
Native/ORT checks cover eleven full streaming clips per model, including both
engines, gains, packed phrase and short tails, with coefficient error≤7.75e-7.
Inactive channels remain zero. No teacher, aligner, new context or extra audio
pass is required in the browser.

Late model SHA256:
`418d03309019e1ecbe6ed0bcac4e812b2c3e978754ca0cb2f8e271858f71d30c`.
Early model SHA256:
`ed85690b0db1df4f78d60149293a0127190a909bd0cf651a7d9bf6cf4deeb79c`.

## Browser speed, memory and complete path

Three fresh desktop WebKit trials per model pass full streaming GPU/native
parity, zeroed/nonfinite-frame checks and byte-identical reset repeats. Maximum
coefficient difference is7.75e-7. Both head arms use the same research runner and
preserved encoder buffer; the production v4 runner is unchanged. On the M5 Max
64GiB, trial long-clip block medians in milliseconds are:

| Model | Trial1 | Trial2 | Trial3 | Median |
|---|---:|---:|---:|---:|
| v4 | 2.52 | .98 | 1.00 | 1.00 |
| Late | 1.50 | 1.04 | 1.08 | 1.08 |
| Early | 2.76 | 1.04 | 1.04 | 1.04 |

Nominal emitted block duration is500ms; early is about481× real time for STA
alone. Both speed gates pass. First-trial variability and these small differences
do not establish a speed advantage. All trials are retained without excluding
the slow first trial. Each head allocates1,598,768 explicit GPU bytes in26 buffers,
versus1,492,236/25 for v4: extra106,532 bytes including the preserved96×101 encoder
activation and enlarged projection uniform. No extra audio pass/session/readback.

Three physical-footprint trials per model sum WebContent+GPU at20ms intervals
with animated James and short/long/repeated accelerated STA. Peaks span
521.88–534.44MiB v4,528.86–532.19MiB late and528.60–531.52MiB early. TTS, initial
rig loading, network and browser parent are excluded. These do not establish
physical-iPhone/full-pipeline memory, or savings from the early feature source.

Verification story: a prerecorded TTS clip is fetched from the local research
page, processed by the selected whole-waveform model/runner, mapped to52 weights
and rendered on James, with matched audio playback and seeking. The original
server had stopped; initial connection refusal is resolved by starting hosted
Vite on its next free port5174 (5173 was occupied), then rerunning the check.

| Boundary | Evidence |
|---|---|
| Asset/runner serving | Source runner HTTP200; ONNX/PCM fixtures fetched successfully |
| PCM → coefficients | Full streaming native/ORT/WebGPU parity on both engines/gains/tails |
| Coefficients → James | Finite tracks, active/inactive channel checks, animated page draws |
| Audio → playback/seek | Sixteen synchronized review cases pass exact frame seeking |
| Errors/idle | No page errors; renderer counters advance after every memory run |

Review is available at `/.local/compact-early-v1/review.html`. This is an automated
rendering/control check, not a blind perceptual study. There is no backend
inference API in this experimental inference path; HTTP serves assets only.

## Decision

Reject this candidate. The earliest learned features provide no practical error
gain for this pointwise correction head and worsen contact metrics. They preserve
the measured phone-event hits, which does not override the framewise failures.
The result does not rule out raw spectral inputs, multiple earlier layers, a
temporal correction branch or every residual architecture. Those are separate
experiments. Keep the published v4, WebGPU default and continuous idle animation.

# Jaw decomposition audit and assistant-teacher pilot

See the [design](compact-assistant-teacher.md) and
[reproduction instructions](../../scripts/research/assistant_teacher/README.md).
Evidence stays ignored under `.local/compact-assistant-v1`. No production
defaults, weights, runtime, idle animation or deployment change in this study.

## Target audit

The fixed sample contains16 original-gain training clips (two per eight voices)
and8 development clips (two per four other voices), selected by stem hash.
The pinned full James teacher is rerun with neutral emotion and seed0. Regenerated
skin-fit weights and raw lip gaps agree with cached labels within5e-5. This checks
the inspected sample, not every corpus label. Jaw-anchor deltas are reconstructed
from the same native Maya lower-denture shapes and rigid-anchor conversion as the
browser. Audit geometry is not saved; sufficient statistics and five-point jaw
tracks are retained. No labels are replaced.

The current solver already uses bounded coefficients, mouth vertex weighting,
ridge regularization and explicit nine-point lip-gap fitting. The missing term
is the separate teacher jaw signal. JawOpen and mouthClose have weighted-skin
basis cosine.8071 and lip-gap basis cosine.9838: lip separation alone gives little
information for distinguishing their contributions. This is overlapping influence,
not proof of nonunique solutions or a demonstrated student-error cause.

The diagnostic adds a soft jaw term inspired by NVIDIA section9.4, with train-only
sigma.33358cm and maximum Hessian trace10% of the existing objective, dynamically
weighted toward small jaw movement with exposed lips. It uses95 bounded FISTA
iterations, no temporal smoothing and no parameter sweep. This is our adaptation,
not a numerical reproduction of NVIDIA's solver.

| Sample / metric | Original fit | Soft jaw constraint |
|---|---:|---:|
| Training jaw-anchor RMS | 1.23384mm | 1.06716mm |
| Development jaw-anchor RMS | 1.19782mm | 1.02172mm |
| Training lip-gap RMS | .36935mm | .37028mm |
| Development lip-gap RMS | .37429mm | .37631mm |
| Training weighted frontal skin RMS | .44660mm | .45551mm |
| Development weighted frontal skin RMS | .43930mm | .44738mm |
| Training contact F1 | .96830 | .96945 |
| Development contact F1 | .96020 | .96451 |
| Training /m,b,p/ hits | 4 / 9 / 27 | 4 / 9 / 27 |
| Development /m,b,p/ hits | 4 / 3 / 4 | 4 / 3 / 4 |

These are voice-macro measurements. Jaw error improves about15% on development,
while skin and gap fit slightly worsen. Small event counts cannot establish broad
closure preservation. Mean coefficient velocity rises.00758→.00785 on training
and.00726→.00750 on development; the constraint is not a smoothing improvement.
The surface residual reconstructs the weighted quadratic in float64, clipping
tiny negative squared residuals from cancellation. No human anatomical ground
truth is supplied by the teacher anchors.

This identifies a real jaw-retargeting tradeoff, not a solution to the student's
multi-millimetre training residual. Target learnability has not been measured by
this audit. Keep old targets for the assistant experiment to isolate its effect.

## Assistant and matched students

The offline assistant widens the eight-block residual network96→192 channels and
adds a96-channel task bottleneck before the existing39-output head. Its1,266,663
parameters and5,994,661-byte waveform ONNX are training machinery only. Initial
widening preserves v4's output within4.18e-7 and bottleneck features within3.82e-6.
Complementary bottleneck-column noise cancels on duplicate features while
breaking gradient symmetry; the encoder-half gradient difference is.00061175.
The tiny export before training is byte-identical to published v4.

An initial trajectory reached export with a checkpoint-loader strictness error.
It and its source/protocol are archived under `export-repair-v1`. Pure duplication
was also recognized as a symmetry trap: a larger parameter count alone would not
ensure independent hidden channels. The function-preserving complementary-column
initialization and its explicit gradient check were then frozen before rerunning
the full assistant. Intermediate preflight artifacts are archived under
`export-repair-v2`. Only the final pilot is reported as the assistant experiment;
these repairs do not constitute a training sweep or extra evaluated candidates.

The final assistant trains7,440 scheduled updates, selects step6200 on development,
and improves combined common/replay development gap RMS2.42181→2.28825mm (5.515%)
with F1.82122→.83173. It passes the predeclared3%/F1 development gate. This gate
does not prove improvement over an equally trained tiny model or held-out quality.
Tiny control and distilled arms then receive identical initialization, seed37,
3,720 scheduled batches, learning-rate schedule and original objective. Treatment
adds training-only normalized bottleneck-feature MSE and assistant-output loss;
the original full-teacher geometry/contact targets remain in both arms.

Assistant SHA256:
`7f993e8548f16f2454743765ab4db3eae5898e7842a95aa021a696cd566e42db`.

## Quality result: reject the tiny distilled model

The regression cohort is the previously inspected24-text/four-voice set,96
original clips plus384 gain cases. It is not fresh confirmation. Original-gain
training fit covers1,280 primary clips; all gains cover3,840 cached cases.
Models select only on the development selector, including initialization. Both
tiny arms select step2976 and share sampler SHA256
`60025a93a676c1dc18ce066fb3f0160484b7e3142796d7d1ab66cc2efb4258b6`.
The ordinary control reproduces the preceding ordinary control byte-for-byte.

| Model | Original training gap RMS | Regression gap RMS | Contact F1 | /m,b,p/ hits |
|---|---:|---:|---:|---|
| Published v4 | 1.83882mm | 2.67389mm | .79595 | 36 / 65 / 55 |
| Offline assistant | 1.56314mm | 2.53327mm | .80060 | 35 / 69 / 54 |
| Matched tiny control | 1.74204mm | 2.62794mm | .79797 | 35 / 64 / 54 |
| Tiny distilled | 1.72965mm | 2.61803mm | .79763 | 33 / 64 / 51 |

The larger assistant is5.259% better than published v4 and3.602% better than
the tiny control on regression RMS. Its extra training budget and architecture
both differ, so this does not isolate capacity alone. Higher framewise F1 and
total158 phone hits still coexist with one fewer /m/ and /p/ hit than v4 and
worse missed-contact rate. It is not a closure-safe browser replacement.

Distillation improves training RMS only.711% and regression RMS.377% over its
matched control. It loses five phone hits, F1 falls slightly, and Supertonic F1
worsens. Phonon gap is2.49109→2.48270mm and F1.80137→.80248; Supertonic gap is
2.76480→2.75336mm and F1.79456→.79278. Those small RMS gains do not satisfy the
predeclared3% extra-gain or closure guards. False-closure rate improves
.07590→.07443 but missed-contact rate worsens.15668→.16058. Conditional onset MAE
worsens16.352→17.350ms, while release MAE improves11.790→11.445ms. Timing is
relative to actual teacher closures, with up to100ms anticipatory association;
read timing alongside lost events.

Historical and feature-regression guards pass; gap, macro/per-engine F1,
per-phone hit, missed-contact and onset guards fail. No new confirmation corpus
is generated for this failed screen. These are one-seed pilot findings and do not
prove that every intermediate-teacher architecture or distillation loss fails.

Control SHA256:
`63ddc0c64851edc445eba1364e8b160a8a680c4d3b66638fc57a17b782cfe867`.
Distilled SHA256:
`ed074f93ba41c1433fb3aa675177f70a07e03b6ea16a66cac02f577ea0ac584a`.
Both tiny models retain331,047 parameters,2,251,827 bytes, the exact original graph
operations and frozen waveform frontend/interpolation/normalization/channel-map
constants. Eleven full-clip native/ORT cases per model, including both engines,
gains, short tails and the packed regression, stay within1.44e-6 coefficients
including the trained assistant. There is no runtime teacher, phoneme aligner,
feature loss or added state in the tiny export.

## Browser checks and decision

Three fresh WebKit parity trials per tiny model use the unchanged production
WebGPU runner, actual animated James, both TTS-engine recordings, four gain
conditions, packed regression and short-tail cases. Reset repeats are byte
identical; no nonfinite, zeroed or missing frames occur. GPU/native reference
differences stay within7.75e-7. Long-clip medians across three trials are1.00ms
v4,.96ms control and.96ms distilled per nominal500ms emitted audio: roughly500×,
521× and521× real time for STA alone on the M5 Max64GiB. The identical operations
and small timing differences do not establish a speed improvement.

All three allocate1,492,236 explicit GPU bytes across25 buffers. Three isolated
physical-footprint trials per model sum WebContent+GPU at20ms intervals with
James continuing to animate through session startup and short/long/repeated
accelerated STA. Peaks across all models span519.85–532.44MiB. TTS, initial rig
loading, network and browser-parent processes are excluded. This is not a
physical-iPhone/full-pipeline result or evidence of a memory improvement.

The sixteen-case synchronized review is available locally at
`/.local/compact-assistant-v1/review.html`; raw teacher geometry gap traces,
teacher-fitted James, v4, matched control and distilled tracks are retained.
The automated audio/seek check is separate from a blind perceptual quality study.

Reject the distilled model and leave published v4/WebGPU/continuous idle intact.
The audit establishes a modest jaw retargeting tradeoff; the assistant establishes
that another offline model can predict lower RMS in this setup. Neither proves
a substantial closure-preserving quality improvement at the current tiny model
budget. Do not claim that the audio frontend is uniquely responsible or that all
assistant distillation methods have been exhausted. No new model is uploaded,
published or deployed.

# Full-corpus training fit and final-layer refit

**The training set is not fitted closely. A final-layer refit lowers RMS but
substantially worsens lip closures, so keep the published model unchanged.**

[Reproduction](../../scripts/research/fit_diagnostic/README.md) ·
[Learnings](learnings.md#41-measure-full-training-fit-and-test-the-final-readout-before-attributing-error-to-generalization).

## Complete training-set measurement

All3,840 primary training clips (751,498 frames) and549 replay clips (99,188
frames) were scored. Training/dev/test remain separate. Primary original-gain
has1,280 clips; all-primary includes original/half/boost. The existing test set
has96 original recordings and384 gain cases. Primary train uses eight voices;
development and test use other voice presets and different text distributions.
The numerical gap cannot by itself identify a pure generalization penalty.

| Model | All primary train RMS mm | Original-gain train RMS mm | Original test RMS mm | Original train F1 | Original test F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published v4 | 1.91351 | 1.83882 | 2.47594 | .85444 | .79904 |
| Ordinary fine-tuned control | 1.80719 | 1.74204 | 2.44256 | .86139 | .80281 |
| SyncNet arm | 1.80916 | 1.74377 | 2.44231 | .86140 | .80209 |
| Frozen-body least-squares head | 1.76196 | 1.69868 | 2.38811 | .84380 | .77553 |

The first three models are the previously locked exports; no retraining or
checkpoint reselection was used to measure their training fit. The new head's
previously evaluated test cohort is **inspected regression evidence**, not a new
independent confirmation.

The same primary training data's teacher-fitted James reference discrepancy is
.36193mm across all gains and.36254mm for original gain. The SyncNet model differs
from the fitted targets by1.81466mm all-primary /1.74879mm original, so its large
training residual is present even against representable fitted targets.
The fitted reference is not a strict lip-gap optimum. Do not subtract these RMS
values or interpret them as independent additive sources of error.

| Separate cohort, original gain | v4 RMS mm | Control RMS mm | SyncNet RMS mm | Head refit RMS mm |
| --- | ---: | ---: | ---: | ---: |
| Historical replay training,453 clips | 1.61074 | 1.51677 | 1.51917 | 1.48097 |
| Extra replay training,96 clips | 1.80740 | 1.68567 | 1.68986 | 1.66261 |
| Common development, original subset | 2.39189 | 2.35469 | 2.35537 | 2.35370 |
| Replay development,29 clips | 2.23096 | 2.19716 | 2.19892 | 2.16832 |

There is both substantial training residual and higher held-out error. This
rules out an exact-fit account of the full training corpus, not one unique cause.
The earlier eight-clip memorisation result establishes tiny-set learnability;
it does not establish adequate capacity, context or optimization across the corpus.
Target ambiguity/conflicting teacher histories have not been quantified here.

## Controlled follow-up: optimal affine readout for fixed features

To test whether a simple decoder recalibration could recover the error, freeze
the ordinary control's encoder, eight temporal blocks, interpolation, frontend
and normalization. Solve only its final39×96 affine readout for normalized
coefficient MSE. This changes3,783 existing values and adds no parameters.

Use all primary/replay training frames, with75% total primary weight divided
equally among eight voices and25% replay weight. Accumulate float64 sufficient
statistics; center and standardize features and solve with fixed eigen cutoff
1e-9. Rank is96/96 and normal-equation relative residual1.32e-15. No development
selection, weight sweep or test tuning is used. This is an optimal-readout
**diagnostic for a specified objective**, not a matched-budget full-network
training comparison, proof about arbitrary nonlinear readouts, or a release claim.

Weighted *unclipped* normalized coefficient MSE falls.257119→.250124 (2.72%).
Playback coefficients are still clipped to[0,1]. Original-gain training lip-gap
RMS improves2.49%, and the inspected test RMS improves2.23% versus control.
However, contact F1 loses2.73 percentage points on test, with these event counts:

| Model | /m/ hits | /b/ hits | /p/ hits | Total |
| --- | ---: | ---: | ---: | ---: |
| Ordinary control | 41 | 67 | 55 | 163 |
| Head refit | 34 | 51 | 32 | 117 |

Phonon test gap improves2.17683→2.11342mm but F1 falls.81287→.80096.
Supertonic gap improves2.70828→2.66280mm while F1 falls.79274→.75009.
Average squared error rewards predictions that compromise important contacts.
This repeats the need to keep closure and timing measures alongside RMS, rather
than accepting a numerical error reduction as a better animation.

The readout's best stated-objective fit remains far from exact, and a simple
linear recalibration does not deliver the desired accurate model. This does not
prove that the audio frontend is the bottleneck, that wider models will fix it,
or that adding data is useless. Learned representation, nonlinear decoding,
available context, label consistency and full-network optimization remain
possible contributors. Test those separately; do not infer a single cause from
the training/test difference or this affine-readout result.

## Integrity, scope and decision

Feature arrays match frozen hashes; source/manifest/audio/label fingerprints
recompute exactly. Reassembled cached coefficient/gap targets match every label
frame. Legacy replay's older hash schemas are handled through their original
cache fingerprints, with every declared waveform hash checked. Fixed waveform
ORT parity covers29 clips per model, max coefficient difference1.13e-6. The new
scoring adapter matches the existing evaluator on768 clips for RMS, F1 and
bilabial hits. Sources/protocols and complete per-voice/condition reports are saved.

Only `student.output.weight` and `student.output.bias` change in the refit.
All331,047 parameters, ONNX operators and other tensors remain identical;
file size is still2,251,827 bytes. There is no new browser speed or memory claim:
this rejected diagnostic was checked by graph equality and native parity, not
new browser benchmarking. No production runtime, WebGPU default, idle animation,
CDN model or Vercel deployment changes. The contact regression is enough to stop
this head from proceeding to release confirmation.

Artifacts remain ignored in `.local/compact-fit-v1`. Initial adapter failures
are retained separately; they did not produce completed measurement reports.
Head model SHA256:
`108c247e3171528f837f92ef51c76d1218f6c479bbe502c4c0954f81f968f2a3`.
The baseline/SyncNet hashes remain those in the preceding SyncNet report.

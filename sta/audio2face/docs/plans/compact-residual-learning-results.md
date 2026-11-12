# Residual learning: structured error, no closure-safe quality win

See the [design](compact-residual-learning.md) and
[reproduction guide](../../scripts/research/residual_learning/README.md).
Evidence stays in ignored `.local/compact-residual-v1`. Keep deployed v4, default
WebGPU and continuous idle animation unchanged. No model is uploaded or deployed.

## What was tested

Published finetune-r4 v4 is frozen, including its39-channel output head and all
audio/temporal layers. Both arms use the same float32 cached96-channel learned
features and39 normalized base predictions, with no new history or lookahead.
They add equal135→96 GELU→39 pointwise heads,16,839 trainable parameters each.
Residual output is added to v4 before final clipping; direct control predicts
the full normalized coefficients. The frozen base is present in both exports.

The direct head initializes paired GELU units to reproduce the base function;
the residual head has zero outgoing weights. Both first layers are identical.
Initial functions/losses and waveform outputs match, but initial gradients differ.
The residual skip supplies a known function outside the trainable head; equal
parameter counts do not prove identical hypothesis classes. This tests a concrete
residual-head design against a matched direct head, not every residual method.

Both arms use the established coefficient/gap/velocity/contact/timing objective
on final predicted weights, with identical seed37,7,440 scheduled batches,
voice-balanced primary/replay sampling, AdamW schedule, clipping and development
selector. No new pure residual-MSE loss or closure adjustment is introduced.
Residual selects step1240; control7440. Their full scheduled sampler hash matches:
`589ce3223580562376a65c7a6dc5876ae0359524f120d90478e167b9f83f44f3`.
Earlier checkpoint selection is allowed by the predeclared shared selector;
exposure is matched, selected training duration is not forced equal.

The only preflight repair adapts frozen-tensor comparison to ONNX's renumbered
anonymous Gather initializer; values/dtypes/shapes are still checked. Old source
and cache protocol are archived in `export-preflight-repair`. Cached feature
contents are unchanged and rehashed. No trained trajectory or loss is changed.

## Is the error noise?

Across751,498 primary training frames (all gain variants), normalized coefficient
errors before clipping have median within-window adjacent-frame correlation
.96044. Top five covariance directions account for51.927% of residual variance.
A constant channel bias accounts for1.235% of normalized squared error.
Statistics are frame weighted and use722,653 valid adjacent pairs; boundaries
between windows are excluded. These are training diagnostics, not voice-macro
test scores or a normality test.

The residual is temporally and cross-channel structured. This alone does not
establish predictability from the available features: teacher history, omitted
information and smooth animation can also produce correlated errors. It does not
identify how much error is intrinsically irreducible.

## Fresh-speech quality

Before scores,24 transcripts (12 targeted /m,b,p/vowel,12 Arctic) are frozen,
excluding2,773 previously planned/observed normalized texts. Four Phonon/Supertonic
presets are outside training. This produces96 original recordings and384
independently teacher-labelled gain cases. Fresh model hashes are locked after
development selection and before scoring. Training fit covers all3,840 primary
gain cases; the table's original subset contains1,280 clips. Old test cohorts
remain separate regression evidence. Do not compare this cohort's absolute
scores directly with previous cohorts' scores.

| Model | Original training RMS | Fresh lip-gap RMS | Contact F1 | /m,b,p/ hits |
|---|---:|---:|---:|---|
| Frozen v4 | 1.83882mm | 2.46765mm | .73837 | 53 / 65 / 51 |
| Direct-head control | 1.81087mm | 2.42992mm | .73903 | 51 / 61 / 48 |
| Residual head | 1.81740mm | 2.44062mm | .73947 | 51 / 64 / 50 |

Residual improves gap RMS1.095% over v4 but is.441% worse than the matched control.
Training RMS is also worse than control. Total phone hits165 improve on control's
160 but fall below v4's169; each phone still loses at least one hit versus v4.
The predeclared≥3% extra gain and closure-preservation requirements fail.

Phonon fresh RMS/F1: control2.27284/.75478, residual2.28533/.75477.
Supertonic: control2.58699/.72329, residual2.59592/.72418. Both engines worsen in
RMS against control; Phonon F1 also falls slightly below the1e-6 guard tolerance.
False-closure rate worsens.09669→.10089, missed-contact rate improves
.21715→.20824. Conditional onset MAE worsens17.251→17.483ms; release MAE improves
12.110→11.868ms. Timing uses actual teacher closure events and up to100ms
anticipatory association, not exact acoustic phoneme boundaries. Read conditional
timing alongside event counts. Historical and feature-regression guards pass.

A post-selection energy diagnostic finds residual removes2.202% of v4's raw
geometry lip-gap squared error and.759% of normalized coefficient squared error
on fresh originals, voice macro. Direct control removes3.053% and.947% respectively.
These are fractions of total v4 error after output clipping, not conventional
mean-centered R-squared or estimates of irreducible noise. They quantify what
these particular trained heads recover; remaining error is not proven random.

This is one training seed and one fresh cohort. It rejects this pilot's release
case, not all correction networks or every representation/head optimization.
No test-driven hyperparameter sweep or model promotion is performed.

## Size and implementation integrity

Both head models contain347,886 parameters and2,321,912 bytes, versus v4's331,047
parameters and2,251,827 bytes. The extra70,085 bytes are3.11% of v4's download;
this is near the original budget, not exactly the same original model size.
Frozen v4 tensors, interpolation/frontend constants and channel map match exactly;
anonymous ONNX Gather constants are matched by value rather than generated name.
Both head graphs have identical operations and equal file sizes. Eleven full
streaming native/ORT cases per model, covering both engines, gains, packed phrase
and short tail, stay within8.35e-7. Inactive channels remain exactly zero.

Control SHA256:
`975d8cd7cf97618fcfa3758c5931f224d02f5dd10bb1cdf7519ec31ae38fb427`.
Residual SHA256:
`c5b9591a3e5c8fec8b6f4c3bcbfe210d221d06d09d335f3d20cc61b00e5783b6`.

The research-only WebGPU runner uses the production audio/convolution kernels,
then one fused workgroup per animation frame for the original projection and
small head. Shared features and base predictions remain on the GPU. No second
session, waveform pass, readback, runtime phoneme aligner or teacher is added.

## Browser speed, memory and review

Three fresh WebKit trials per model pass complete streaming GPU/native parity,
finite/zeroed-frame checks and byte-identical reset repeats across both engines,
gain variants, short tails and the packed-phrase regression. The initial-function
browser controls also pass. Benchmark training and teacher labelling have
finished before these isolated trials.

On the M5 Max64GiB, median long-clip block times across three trials are.96ms
v4,1.00ms direct control and1.02ms residual per nominal500ms emitted audio.
STA alone is approximately521×,500× and490× real time. Residual is6.25% slower
than v4 and2% slower than control by these medians, within both predeclared
speed limits; trial values and small differences should not be overinterpreted.
Both head arms allocate1,559,980 explicit GPU bytes across25 buffers, versus
v4's1,492,236 across25: extra67,744 bytes including the enlarged projection
uniform. No extra waveform/activation session is required.

Three physical-footprint trials per model sum new WebContent+GPU processes at20ms
intervals with James continuously animating through startup, short/long and
three accelerated repeats. Peaks are523.56–532.94MiB for v4,526.22–537.31MiB for
control and528.11–538.52MiB for residual. These exclude TTS, initial rig loading,
network and browser parent. They do not establish full-pipeline/iPhone memory
use or a process-memory saving. The scripted idle rendering checks pass.

The sixteen-case synchronized teacher/v4/control/residual review passes audio
playback and exact frame seeking; it is available at
`/.local/compact-residual-v1/review.html`. This is an automated rendering/control
check, not a blind perceptual quality study.

## Decision

Reject this candidate for deployment. It meets the small-file/runtime budget but
fails the extra-error-gain and closure-preservation requirements. Error has clear
temporal/cross-channel structure, yet this particular correction head predicts
little of it from the frozen late features. Do not infer that the rest is random
noise, that the original audio frontend is uniquely at fault, or that all residual
architectures fail. Earlier/shared spectral features, different context or better
optimization remain separate hypotheses; none is demonstrated by this pilot.
Keep the published v4 and continuously animated James unchanged.

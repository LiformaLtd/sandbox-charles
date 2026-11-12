# Jaw-target audit and intermediate-teacher pilot

Frozen design, 2026-11-08. Keep production v4, WebGPU and idle animation intact.
No generated models or teacher geometry enter Git. Final student must retain the
331,047 parameters and 2,251,827-byte waveform ONNX graph, including constants.

## 1. Audit before training

Use two deterministic original-gain clips per voice from old-train and two per
voice from common-dev, chosen by hash, never by measured error. Regenerate full
Audio2Face James geometry with the existing pinned model, neutral emotion and
seed0. Check cached skin-fit weights and raw lip gaps against regeneration.
Use the same Maya lower-denture blendshapes and rigid-anchor conversion used by
the browser to reconstruct the five jaw anchors from fitted coefficients.

Measure jaw-anchor RMS, raw-geometry lip-gap RMS, framewise contacts and /m,b,p/
event hits, coefficient temporal variation, and jawOpen/mouthClose basis
correlation. Compute the existing weighted skin/ridge/gap quadratic objective.
Audit a separate soft jaw term following NVIDIA section9.4: train-only median jaw
displacement sets sigma; fixed maximum jaw Hessian trace is10% of the original
objective trace, modulated by small jaw movement and large central lip opening.
Use the same95 bounded FISTA iterations, with framewise Hessian and step size.
This is an exploratory diagnostic, not an exact reproduction of NVIDIA's solver.
Do not overwrite labels. Do not claim a target improvement if original geometry
or contact metrics deteriorate. No student training on altered targets in this
pilot, so target changes and assistant effects remain separate experiments.

## 2. Intermediate teacher and tiny student

Use the existing full primary/replay corpus and development selector. Initialize
from published v4 finetune-r4. The assistant has192 channels, the same eight
residual dilations and frontend/context, plus a96-channel task bottleneck before
the existing39-output head. Duplicate the96-channel model with correctly scaled
incoming weights and an averaging bottleneck so its initial function matches v4.
Complementary zero-sum bottleneck-column noise (seed37, standard deviation
.01/sqrt(96)) preserves that function while breaking duplicate-channel gradient
symmetry. Verify this explicitly before training.
This larger model is offline only. Train for7,440 updates with the established
shape/gap/velocity/contact/timing objective and voice-balanced batches. Select by
the existing development macro gap RMS+.25*(1−F1), including initialization.

Before tiny-student training, require assistant development gap RMS to improve
at least3% over v4 with no macro contact-F1 loss; otherwise stop the assistant
route and report the failure. This is a pilot gate, not independent confirmation.
If it passes, train tiny control and distilled arms from identical v4 weights
for3,720 updates, identical seed37, scheduled batches and AdamW schedule. Control
uses the established objective. Treatment adds .1 normalized bottleneck feature
MSE and .25 normalized SmoothL1 assistant-output distillation. Assistant features
are on the identical60fps timeline; valid central30-frame crops only. Feature
scales are frozen from fixed training batches, not development or test data.
The frozen assistant sees the same input and context. All distillation machinery
is removed on export. There is no runtime phoneme model or added browser state.

## 3. Evaluation and safeguards

Lock sources, input model and dataset/cache hashes before training. Select only
on development data. The previously inspected closure-protected confirmation
cohort and historical cohorts are explicitly regression evidence, not fresh test
claims. Score published v4, assistant, matched tiny control and distilled student
with the established full-clip streaming crops and anticipatory closure metric
(up to100ms before /b,p/ onset). Report original-gain training RMS separately
from held-out regression RMS, per-engine F1 and /m,b,p/ hits. Require at least3%
extra gap improvement over matched control without macro/per-engine F1 loss or
per-phone hit loss, and preserve published-v4 contacts too. Report false closures,
missed contacts and onset/release errors; no more than .1ms release/onset worsening
or1e-6 rate worsening against control. No promotion without a subsequent fresh
confirmation cohort and browser/physical-iPhone review.

Validate exact final graph/constant/size equality and MLX/ORT coefficient parity.
Measure actual production WebGPU speed and explicit GPU buffers with the animated
James page. Desktop WebKit footprint measurements, if made, exclude TTS and are
not a physical-iPhone claim. Do not publish or deploy this pilot automatically.

Research basis: [NVIDIA sections9.3–9.4](https://arxiv.org/html/2508.16401v1)
and [EA hybrid distillation](https://arxiv.org/html/2507.18352v1). This is an
adaptation to our existing model family, not reproduction of their architectures,
data volumes, thresholds or published metrics.

# Teacher-feature distillation: expanded-corpus results

**Normalized teacher-feature supervision did not meaningfully improve lip sync.**
The distilled export has 0.0126% lower original-level RMS than its matched
fine-tuning control (0.000306 mm), with slightly lower contact F1 and three fewer
well-timed closures. Keep current published v4 scale-r3. No model was uploaded
or deployed from this experiment.

The [frozen experiment](compact-feature-distillation.md) fixes the hypothesis,
loss, data, updates, selection and 3% screen before optimization. This revisits
the earlier inconclusive small-corpus feature trial; it is not a first test of
the idea. See [reproduction steps](../../scripts/research/feature_distillation/README.md).

## Controlled comparison

Both arms initialize from current v4 scale-r3, SHA
`407739196cd98c89ec619b6af527f1021a3702cf8b1213c006f86408d1701137`.
Use the existing expanded training set: 3,328 primary original recordings /
3,840 gain-labelled examples plus 549 unchanged clean historical replay clips.
The model's normalization, frontend, 96 channels, eight residual blocks, 39
active/52 exported coefficients and animation objective stay fixed.

Both instantiate the same training-only affine 96→768 head and share identical
seed-37 initialization and sampled primary/replay minibatches. Distillation adds
valid-frame 1−cosine similarity against the teacher's last audio-encoder layer,
weighted 0.1 and ramped over 372 updates. Control weight is0. Historical replay
has animation supervision only. The head is discarded before ONNX export.
Teacher features use exactly the original waveform, one-second window, 60fps
interpolation and emitted crop; no larger audio context is added to the student.

The 28,845 primary hint windows are offline f16 memory maps, checked against
cached geometry targets and validity masks. CPU/MLX auxiliary loss matched
exactly at initialization; gradient relative L2 difference was 3.85e-7. Hint
backbone gradient norm was 0.2272; the zero-weight control has zero hint gradients.
Three tests cover padded-frame exclusion/gradients, feature-scale invariance
and finite zero-valid-frame handling. TF32 is disabled before MLX import.

Each arm trains 3,720 updates with 48 voice-balanced primary + 16 replay windows,
identical AdamW/cosine schedule, clean replay, development/held-out-voice guard
and source/cache locks. The complete sampler hashes match. Development-only
selection chose step**2976** for both arms. Initial development outputs/scores
match exactly. Checkpoints retain optimizer/RNG and best student weights.

The head learned its auxiliary task: training-batch cosine loss fell from 0.750
at step 124 to 0.382 at step 3720 (control stayed about 1). Improving this internal
feature reconstruction did not translate into extra animation accuracy.

## Fresh lip-sync evaluation

24 new transcripts (12 authored articulation/vowel tests and 12 broad unchanged
CMU ARCTIC sentences) × four previously studied training-held-out presets =
**96 new recordings /384 independently teacher-labelled gain cases**, about
310.4 seconds of original speech. Engines and catalogue genders are balanced.
The audit excluded 2,453 retained prior normalized texts, including the scale
study's training/dev/test. The presets themselves are not new voices.

Each changed-gain waveform has its own teacher inference/fitting. Instrumented
and untouched full-teacher geometry matched exactly over 3,997,395 values.
The source text hash/commit and CMU license remain in prior data-scale provenance;
no human recordings were used. Visual cases and evaluator were fixed before
fresh scoring. Both exported checkpoints were locked before reading scores.

| Model | Original lip-gap RMS mm ↓ | Contact F1 ↑ | Opening RMS mm ↓ | Well-timed /m b p/ hits ↑ |
|---|---:|---:|---:|---:|
| Current published v4 scale-r3 | 2.48979 | 0.79312 | 2.85016 | 152 |
| Matched fine-tuning control | 2.42607 | 0.80250 | 2.77146 | 156 |
| Teacher-feature distillation | 2.42576 | 0.80228 | 2.77098 | 153 |

These are macro-voice metrics on this **new cohort**. Current v4's 2.48979 mm here
is not a revision to its previously reported 2.72455 mm on the older 352 recordings.
Directly comparing those numbers would mix test distributions.

The control improves 2.5595% versus current v4; distillation improves 2.5718%.
Thus essentially all improvement is shared fine-tuning, not evidence for feature
supervision. Control-relative changes by engine are +0.0599% Phonon and−0.0279%
Supertonic: effectively tied, with the latter failing the no-regression guard.

| Gain condition | Current v4 RMS mm | Control RMS mm | Distillation RMS mm |
|---|---:|---:|---:|
| Original | 2.48979 | 2.42607 | 2.42576 |
| Half | 2.60384 | 2.50519 | 2.50287 |
| Headroom boost | 2.55343 | 2.49995 | 2.50030 |
| Quarter | 3.05351 | 2.91893 | 2.91241 |

Well-timed /m/, /b/, /p/ counts are 49/55/48 for v4, 48/58/50 for the control and
48/56/49 for distillation. The evaluator retains anticipatory teacher-contact
association and the existing one-to-one timing rules; no threshold was changed
to improve these results. Full per-phone onset/release errors, association
ambiguities, engine/category/voice scores and opening/velocity metrics are in
the JSON reports. Timing errors are conditional on matched events, so they must
be read alongside missed-event counts.

Historical original RMS/F1: v4 2.61816/.76861; control 2.54635/.77603;
distillation 2.54473/.77782. Inspected scale-regression RMS/F1: v4
2.72455/.78855; control 2.64753/.79642; distillation 2.64612/.79692.
Those guards pass; they are regression checks, not fresh confirmation.

The predeclared screen fails primary RMS gain, engine non-regression, contact F1
and total well-timed closures. Gains, per-phone allowance and historical/scale
guards pass. Keep `promising=false`. This one fixed coefficient/seed/layer trial
does not rule out every form of distillation, but it provides no reason to add
this method or tune it against the inspected test.

## Browser speed and memory

All three exports retain **331,047 parameters  / 2,251,827 bytes**, identical ONNX
operator topology, **25 buffers  / 1,492,236 explicit GPU bytes** and two reusable
compute pipelines. The auxiliary head is absent from the exported initializers.

Twelve complete native-ORT/WebGPU track fixtures cover both engines, four gain
levels, female voices and the complete packed phrase. Three fresh settled
WebKit runs plus first-clip repeats produced 117 comparisons. Maximum coefficient
difference was 7.01e-7; no nonfinite or unexpected zeroed frames were found.
Repeated first-clip GPU results were byte-identical for every model. A sixteen-
case four-lane James viewer passed synchronized audio and exact seek checks.
Representative screenshots are inspection aids, not a blinded perceptual study.

| Model | Warm block median ms | Observed packed medians range ms | Sampled WebContent+GPU peak MiB |
|---|---:|---:|---:|
| Current v4 | 1.02 | 0.98–1.06 | 532.5 |
| Control | 1.06 | 1.00–2.10 | 524.6 |
| Distillation | 1.01 | 0.98–1.10 | 535.0 |

Warm timing is the median of six packed-track medians per model across three
fresh runs, after rig startup and five warm-up calls. A full block emits 500 ms
of animation audio. Timing variability does not establish a weight-dependent
speed change for this unchanged graph. These are desktop M5 Max measurements,
not iPhone or TTS click-to-first-audio timings.

Physical footprints are concurrent sums of newly launched desktop WebKit
WebContent+GPU processes sampled every 20 ms, with normal James idle animation
active throughout short, long and repeated long prerecorded STA. **TTS is not
loaded.** Differences of this scale can arise from driver/process variability; these
measurements do not establish a memory saving or a meaningful regression.
Browser/network processes are excluded and brief peaks can be missed. There is
no full-pipeline or physical-iPhone under-2GB claim.

## Artifacts and decision

Ignored root: `.local/compact-feature-v1`. It retains corpus/teacher provenance,
hint maps, CPU-gradient checks, frozen sources, training/selection locks,
optimizer/RNG checkpoints, predictions, full evaluation reports, browser timings,
process traces and synchronized `review.html`. The research source server used
http://127.0.0.1:5181/.local/compact-feature-v1/review.html; use the current free
port when restarting Vite.

Training protocol SHA: `b2f7b1882387aec6ecd2c65d3039914cbbb7c995deba4a3789bff5c41f2a9260`.
Control export: `ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`.
Distilled export: `e3a2dc9de3aa8fa8800ca1144c2d3d74dbd646664619c3944943d94ec68496f1`.

Keep deployed v4 unchanged. Retain the useful finding that additional ordinary
fine-tuning improved both fresh and inspected regression sets, but do not
attribute it to feature matching. Further experiments should target the actual
rendered-mouth residuals or difficult speech coverage independently; do not
combine this unsuccessful auxiliary loss merely because it costs no inference.

## Requested ordinary-control release

After reviewing the negative feature-distillation result, the user requested
publishing the **ordinary fine-tuned control** as v4 finetune-r4. It is seed37,
development-selected step2976, SHA
`ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`.
The immutable CDN prefix is `models/speech-to-animation/lf-sta/arkit-student-v4-finetune-r4`.
The publisher verifies downloaded checksums for the model, licenses and manifest;
scale-r3 remains available at its preceding immutable URL. This is a requested
release for testing, not a reinterpretation of feature distillation as useful.
The hosted v4 redirect selects this control while keeping WebGPU as the default
and retaining continuous idle motion. Physical-iPhone stability remains unverified.

Release commit `78d8e31` was staged as deployment
`dpl_53xPsaEsLQBNCZvzFbJDL6jx2T9D`, verified in desktop WebKit with all three TTS
engines, exact CDN/model checksum, v4/WebGPU/Phonon defaults and animated idle,
then promoted to https://james-audio2face.vercel.app/. Local v4 weights also
match finetune-r4. The preceding feature/distillation research artifacts remain
immutable. These browser checks do not replace physical-iPhone testing.

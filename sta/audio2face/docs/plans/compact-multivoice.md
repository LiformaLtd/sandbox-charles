# Compact multi-voice speech-to-animation study

## Objective and deployment contract

Train a voice-independent English speech-to-James-ARKit model for Phonon and
Supertonic 3. Male/female voices, pitch ranges and UK/US Phonon presets must be
represented. Unseen speakers and unseen text are evaluation requirements.

Keep released v1/v2/v3 and the geometry teacher unchanged. Work is on
`research/compact-multivoice` in the normal checkout, as requested in this chat.
Research artifacts stay in ignored `.local/compact-multivoice`; weights, audio
and label archives must not enter Git. No deployment or baseline replacement is
implied by a promising pilot result.

The first three arms preserve the existing 96-channel network, FFT/log-mel
frontend, one-second input, centre-30-frame streaming crop and dedicated WebGPU
runner. There is no runtime text input, forced aligner, extra speech encoder,
phoneme head or feature projection. James must continue head motion and blinking
when idle. Expected inference memory and arithmetic are unchanged; verify actual
export/runtime parity, speed and full-page memory rather than infer them from
parameter count alone.

## Why change direction

The previous 48-clip Easton confirmation set measured raw-teacher lip-gap RMS
0.0378 cm after fitting the existing rig, versus 0.2580 cm for released compact
v3 and about 0.2517 cm for the longer-trained folded candidate. The rig has much
more lip accuracy available than the student predicts. These are teacher
agreement metrics, not human motion ground truth or overall perceptual scores.

The width experiment copied existing channels; their learned replicas remained
almost identical. It was not a convincing test of a larger independent
representation. Recent data emphasized sentence coverage in one Phonon voice,
Easton. Broad voice/engine generalization was not established. Earlier label
fitters also produced inconsistent coefficient supervision. This pilot creates
fresh, consistently fitted labels for every voice recording instead of replaying
mixed-fit legacy labels.

## Frozen pilot

Definitions live in `scripts/research/multivoice/{voices,prompts}.json`.

| Partition | Phonon voices | Supertonic 3 voices | Texts per voice | Clips |
| --- | --- | --- | ---: | ---: |
| Training | Easton, Alfie, Zoey, Maeve | M1, M2, F1, F2 | 32 | 256 |
| Validation | Micah, Juniper | M4, F4 | 8 | 32 |
| Test | Hugh, Freya | M5, F5 | 12 | 48 |

Every partition is balanced across engines and catalogue gender. Training has
US/UK Phonon voices of both genders. Gender labels describe catalogue presets;
they do not substitute for measuring actual acoustic pitch ranges. Supertonic
locale is simply English, without inventing accent labels.

Texts cover `/m/`, `/b/`, `/p/`, contrasts, broad vowel openings, rounding,
fricatives, conversation and pauses. Identical prompts across voices are kept
within a partition. Normalized texts are checked against all available historical
compact manifests and the earlier research confirmation set. English dictionary
coverage is checked before synthesis. This 52-text pilot is a mechanism test,
not a sufficient final voice-independent corpus or multilingual evaluation.

Assets are pinned to the demo's model revisions. Synthesis uses Phonon 0.6.0
Q8 raw WASM, temperature 0.7; Supertonic uses the demo's JS helper, WASM ONNX
Runtime 1.31.1, eight denoising steps and speed 1.05. Per-recording random seeds
derive from voice identity and text; Supertonic's noise PRNG is explicitly seeded.
Both engines use the demo's resampler to 16 kHz. Cached audio is checked by
recipe and byte hash, never silently reused after changing a recipe.

`prepare.py` records asset hashes and snapshots the optimized TinyAligner Fast
frontend/CTC, Supertonic helper, resampler, student architecture and lip-aware
fitter. Other concurrent projects' source files remain untouched. The aligner
uses its full original graph and dictionary with the optimized frontend/CTC;
this study does not use the experimental English-only projection graph.

Teacher labels use the pinned full model, James identity, neutral emotions,
seed zero and MLX TF32 disabled. Record labeler, teacher source, rig and audio
hashes. Save 52 fitted coefficients plus nine raw geometry lip gaps per frame.

## Experiments, one variable at a time

All first-stage arms initialize from released v3, retain its output normalization,
use exactly the same new data, 64-window minibatches (eight windows per training
voice), optimizer, learning-rate schedule, seed and 24-epoch budget. Every clip
gets its own teacher output; another voice's coefficients are never reused.

1. **Control:** consistent multi-voice data and the existing geometry/contact
   training objective. This isolates the effect of training on the target engines
   and broader voices from auxiliary supervision.
2. **Phoneme supervision:** control plus a training-only linear phone head from
   the existing temporal representation. TinyAligner supplies timed ARPABET
   labels; unaligned gaps/padding have ignore labels. Cross-entropy weight 0.1.
   Alignment labels are estimates, not reviewed phonetic ground truth.
3. **Speech-feature distillation:** control plus a training-only 96-to-768 linear
   projection to the full teacher's last audio-encoder layer. Layer-normalize each
   frame and use SmoothL1 weight 0.1. The existing teacher already contains the
   relevant encoder; FaceFormer and a separate pretrained model are unnecessary
   for this first distillation experiment.
4. **Combination:** only after inspecting independent validation results, combine
   useful supervision paths under a predeclared recipe. Do not pick a combination
   or weights by repeatedly looking at the same test set.

An instrumented *offline copy* of the native teacher backend exposes
`/model/audio_model/encoder/layers.11/final_layer_norm/Add_1_output_0`, shape
`[1,49,768]`. The original teacher files are not edited. The feature convolution
has a 400-sample receptive field and 320-sample hop. Features are linearly
interpolated from their 199.5-sample centre onto each input window's 60fps sample
grid. Save only the valid centre crop, in float16, alongside each label track.
The training projection is discarded before exporting the student.

Instrumentation was checked against the untouched teacher over 3,997,395
geometry values on a short clip, with maximum absolute difference **zero**.
Both runs used identical PCM/noise/state scheduling. This verifies that exposing
features did not change that reference; it is not a quality result for a student.

Checkpoint selection minimizes mean per-validation-voice
`gap_RMSE + 0.25 * (1 - contact_F1)`. Test labels never participate in training or
selection. One-seed pilot results require independent-seed confirmation before
making a release claim. Evaluation reports per-engine, gender, voice and prompt
category; macro voice scores prevent long clips dominating the headline result.

## Scale-up and structural follow-ups

If either auxiliary task moves validation quality materially, scale the balanced
corpus to hundreds of distinct prompts across more voices before fine-tuning
loss weights. Cover ordinary dialogue, closures, large vowel openings, speaking
rates and expressive delivery. Acoustic variation matters beyond phoneme-pair
text coverage. Regenerate teacher labels after changing a recording. Keep text
and voice assignments independent and freeze fresh confirmation voices/prompts
before selecting a final model.

Compare cleaned legacy-target replay in its own arm after refitting the same
recordings with preserved teacher precision/noise/timestamp settings. Do not
attribute differences between dissimilar old/new corpora solely to the fitter.

If representation distillation still plateaus, test a genuinely independent
larger teacher/student architecture, rather than nearly identical channel
replicas. Then test articulation-state heads or a learned short-motion prior to
represent contact onset, hold and release. These are later architectural
experiments, not assumed winners. Permit a modestly larger deployed model only
if it improves quality while preserving mobile WebGPU peak memory and latency.

## Execution and acceptance

See `scripts/research/multivoice/README.md` for reproducible commands. Report
teacher fitting quality, coefficient error, lip-gap RMS, contact precision/recall,
`/m b p/` event hits and onset/release timing. Compare v3 and the three pilot arms
on the same untouched test audio. Retain the historical Easton confirmation set
as a regression test, not the sole release target.

Before release: independently seeded repeats; reviewed synchronized James mouth
comparisons; numerical CPU/WebGPU parity; warm block latency; short and long
full-page peak memory; repeated playback; idle animation; actual iPhone testing.
Desktop WebKit physical-footprint measurements are not iPhone memory guarantees.

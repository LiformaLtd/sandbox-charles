# Expressive ARKit model: first candidate and James review

Date: 2026-11-11. Issue: LIFORMA-107. Branch: `research/expressive-arkit`.

## Status

A new research candidate has been trained and exported: **2,781,197 bytes**,
52 facial coefficients, explicit emotion and gaze-style controls, plus a separate
unchanged articulation track. It runs in the browser's ONNX CPU runtime in a
separate James 3D comparison view. It has **not** passed the gate for production
or for a claim of fully validated ARKit visual expressivity.

The released finetune-r4 model, primary James app, Player, v1 textures and 2.5D
mapping are untouched. There has been no model publication or Player integration.

Open `http://127.0.0.1:5173/expressive.html` while the research server is running.
The left face uses current v4, and the right uses the candidate. Head motion and
procedural blinks are off, so they cannot be mistaken for learned model output.

## What was built

- Frozen finetune-r4 speech core: 331,047 predictor parameters; original SHA-256
  `ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`.
- New temporal expression adapter: 103,540 parameters, including eye outputs
  superseded by the separate gaze branch in the final graph.
- New gaze branch: 23,812 parameters, conditioned on the same audio features,
  emotion and an explicit source-identity style. This is a gaze prediction branch,
  not a head-rotation predictor.
- 52 bounded facial outputs. Previously learned coefficients and newly enabled
  lateral jaw/mouth coefficients are protected at neutral. Expression targets
  use paired teacher differences relative to neutral on those 43 channels.
- A separate `articulation` output exposes the frozen speech coefficients for a
  future contact compositor. It does **not** automatically constrain the current
  expressive mesh's lips.

The graph accepts `pcm [1,16000]`, `emotion [1,60,10]` and
`gaze_style [1,60,3]`. Outputs are `blendshapes [1,60,52]` and
`articulation [1,60,52]`. The source geometry and frame alignment remain at 60 fps.

Candidate SHA-256:
`ae6a8a30a1338fb8199716dcba4c9bc98234b07404e0be934bc09627ba0926c8`.

This is not a drop-in model for the existing custom WebGPU weight loader. The
review uses ONNX Runtime WASM explicitly; it does not silently switch runtimes.

## Teacher and supervision

The official NVIDIA Multi v3.2 ONNX teacher outputs 88,831 values per frame:
72,006 skin values, 16,806 tongue values, 15 jaw values and four eye rotations.
It returns geometry and rotations, rather than a ready-made 52-channel ARKit
capture stream. The earlier fitting stage enabled only 39 facial channels.

We reran 48 existing Phonon/Supertonic recordings under neutral and all ten
NVIDIA emotion controls. There are 32 training recordings across eight voices,
eight development recordings across four voices, and eight held-out recordings
across four voices. Prompts and voices do not cross the pilot's splits. These
recordings were used in previous research: they are not fresh independent
release confirmation. Repeated emotion runs are not distinct speech recordings.

James supplies the facial expression targets. Its gaze did not exercise
`eyeLookOutRight` in the pilot, so we also generated the official teacher's Claire
and Mark gaze outputs. All eight gaze directions are then exercised without
averaging opposite directions across identities. The identities are explicit
gaze-style controls; the facial geometry remains James. There are 1,584 gaze
sequences: 48 recordings × 11 conditions × three identities.

Labels retain all 52 derived coefficients, the four original eye rotations,
16 fitted tongue controls, 15 jaw values and skin-fit residuals. Raw high-volume
geometry is consumed in memory rather than duplicated on disk.

Emotion is an explicit input. Identical audio deliberately has ten different
expression labels, so this experiment cannot infer which emotion was intended
from audio alone. It learns audio-correlated motion within a supplied condition;
it does not include an audio-emotion classifier. Intermediate intensities and
transitions need additional validation.

## Retargeting calibrations

The facial fit now solves 43 skin channels. Eye rotations drive the eight gaze
channels with a declared endpoint of 25 local degrees. This is a research
calibration, not an Apple or NVIDIA-specified ARKit endpoint. Opposed directions
are mutually exclusive in the gaze decoder.

The supplied skin `tongueOut` target is effectively empty. The candidate currently
uses the canonical `tongueStretch` shape as a protrusion proxy after removing
authoring mouth/jaw movement. **The James visual test shows that this can produce
protrusion during ordinary speech. It is not yet a correct semantic tongueOut
calibration.** A full 52-slot output must not be confused with full validation of
every output's meaning.

The first partial label run used a fixed ridge that suppressed weak eyelid
targets by 22–32% even on synthetic known poses. It was stopped before training.
The replacement uses column-energy-relative regularization. Its 52 one-target
synthetic recovery checks have maximum coefficient error 0.00601. This tests the
solver's self-consistency, not independent semantic accuracy or visual quality.

## Checks and findings

- Nine contract tests pass: gaze axes, exclusion and bounds; bounded fitting;
  neutral protection; output-head gradients; invalid audio/control rejection.
- ONNX graph checking passes. Thirty-six native/ONNX export cases have maximum
  coefficient difference `6.56e-7`. The articulation stream matches the frozen
  source model within the export tolerance for all tested controls.
- All 52 trained outputs vary on the held-out pilot conditions. The smallest
  observed output range is about 0.062. This proves neither complete 0–1 coverage
  nor independently calibrated full-range motion for every channel.
- Combined coefficient RMS is 0.04817, compared with 0.05630 for a simple constant
  expression/style-pose control: a 14.45% reduction. These are hybrid
  teacher-difference targets and numerical fit measurements, not visual release
  confirmation. Gaze's mean held-out signed-angle RMS is about 1.16 degrees.
- The exact appointment regression produces 308 browser frames. The neutral
  expressive output retains the existing mouth behavior. The separate
  articulation output matches v4 exactly in the checked browser runs.
- **The unconstrained expressive output loses baseline contact frames at full
  emotion strength:** 20 for amazement, eight for anger, 15 for joy on the
  appointment sentence. Anger also adds one contact frame. The separate
  articulation stream has zero changed contact frames. This is a known issue,
  not a successful expressive-mouth regression result.
- Browser playback also passed on a second Phonon voice (207 frames) and an
  uploaded Supertonic sample (216 frames). At audio time 0.110298 s, both views
  showed frame 6; after scrubbing to 1.795581 s, both showed frame 107; resumed
  playback at 1.945025 s showed frame 116 in both views.
- Manual rig inspection shows the new outward eye rotation and tongue-protrusion
  target visually. The manual target inspector is explicitly separate from model
  inference. Brows are visibly more expressive under supplied emotion controls.
- Fifty-three protected primary model/runtime files remain unchanged. Frozen
  face-training and gaze-training sources still match their recorded hashes.

The first browser check found a Vite error from forcing an Emscripten module
import out of `public/`. Removing that override and using the installed ONNX
Runtime bundled loader fixed it. The successful playback checks were performed
after that fix. No error or warning was hidden.

## Why head rotation is deferred

NVIDIA demonstrates head motion in
[paper section 9.2](https://arxiv.org/html/2508.16401v1#S9.SS2), using an extension
with two joints' rotations and root translation. The
[official release index](https://github.com/NVIDIA/Audio2Face-3D/blob/release/README.md)
links the same [published checkpoint](https://huggingface.co/nvidia/Audio2Face-3D-v3.0/tree/b74132732fd9a9d29b237bec193ded64c9745e91)
already used locally. Its interface has no head-motion output; the published
[SDK output layout](https://github.com/NVIDIA/Audio2Face-3D-SDK/blob/main/docs/README.md)
lists skin, tongue, jaw and eyes. No downloadable checkpoint exposing that head
extension was found in the official releases checked here.

This is a checkpoint-availability limitation, not evidence that a Mac cannot
run such a teacher. If NVIDIA provides the extension's weights, test its operators
on the local MLX/ONNX setup before deciding on hardware. Per user preference,
head training is deferred and **no DAWN-generated labels are used**.

## Next gates before changing the Player

1. Review the James comparison, including the appointment sentence, across
   emotion strengths. Inspect every channel's semantic motion and gaze signs.
2. Fix `tongueOut` supervision using actual protrusion relative to the moving lip
   boundary. Internal tongue deformation is not equivalent to sticking the tongue
   out; preserve the separate tongue controls for a richer 3D preview if needed.
3. Train or constrain combined expression/articulation to preserve bilabial
   contact. The frozen articulation stream provides the reference. Avoid fading
   the whole expression away on every bilabial, which could introduce pulsing.
4. Verify fresh speech, expressive contact duration, temporal joins, intermediate
   intensities and transitions. Extend the browser WebGPU runner and verify mobile
   performance only after the geometry checks pass.
5. Extend avatar-specific 2.5D calibration in isolation. Existing viseme meshes do
   not automatically supply brow, eyeball or tongue targets, or expression
   textures. Keep current v1 geometry, closure handling, timing and phoneme-driven
   texture blending in place until the new composition passes its gates.

## Run the isolated review

From this task worktree's `sta/audio2face` app, use the explicit source-path
environment variables documented in
`scripts/research/expressive_arkit/README.md`, then:

```sh
npm ci
python scripts/research/expressive_arkit/prepare_review.py
npm run dev:expressive
```

Vite starts on 5173 and tries the next free port if occupied. This research
configuration redirects `/` to `/expressive.html`; normal `npm run dev` keeps its
original entry point. `npm run build:expressive` builds to `dist-expressive/`.
All staged models, meshes and audio stay ignored under `public/assets/`.

The review supports recorded speech, uploaded audio, emotion/gaze-style controls,
two synchronized faces, camera controls and a clearly labelled manual target
inspector. No new TTS endpoint or production service is required.

## Revised neutral-performance acceptance criterion

The user has replaced unchanged mouth coefficients with comparable lip-gap RMS.
The full neutral face should learn the teacher's motion, including brows,
eyelids, cheek/nose and mouth channels. The existing compact predictor can remain
a useful prior without forcing its neutral outputs at inference. Gaze retains
its existing teacher-trained branch; official head motion remains deferred.

Use absolute teacher targets, a slower brow decoder and direct raw-geometry lip
supervision. Select checkpoints only when each neutral validation cohort's macro
lip-gap RMS remains within 5% of the compact baseline. Then check all three James
phrases separately at neutral and full anger against the matched previous model,
using the same nine lip pairs and raw teacher geometry. Report contact F1,
brow fidelity, slow motion and fast jitter separately, so an aggregate RMS does
not hide weaker closures or flattened expression. A passing RMS alone does not
establish full naturalness or production readiness.

The first full-neutral pass failed the appointment phrase gates despite passing
broader cohort averages. It was not promoted. The revised pass increases lip-gap
supervision and replay instead of restoring an exact-mouth neutral gate. The
browser now supports checking the active face's lip-gap RMS; its auxiliary frozen
speech output is only a core-integrity reference.


Final refinement outcome: all three neutral phrase RMS checks pass (+2.99%, -9.16%, -16.03%). Appointment at full anger remains +5.47%, narrowly above the 5% working tolerance, so the candidate was not staged. Strong-anger brow jitter fell 90.9–94.7%, but some slow neutral brow gestures remain too weak. A discarded run with an incorrect loop length was rerun using the correct 1,200-step schedule. Player and 2.5D mapping remain unchanged.

User-authorized visual review: James now serves expressive-full-neutral-v3 (SHA 631b4d010f2f), despite the +5.5% appointment/full-anger lip RMS result. Research-only staging displays that result and allows playback above tolerance. Strict verified staging still rejects failures. Player and 2.5D mapping remain unchanged. Browser confirmed 308 frames, active lip RMS 3.269 mm against the 3.254 mm working limit, with successful playback. Research build passed and all 53 protected files remain unchanged.


Eyelid follow-up: confirmed excess squint/widening/left-blink oscillation in the model. `train_eyelids.py` fits separate six-Hz squint/wide and twelve-Hz blink decoders against absolute teacher labels; parent mouth/brow/gaze/other channels are frozen. `diagnose_eyelids.py` compares raw teacher, fitted teacher and model geometry; `verify_eyelids.py` checks held-out lid accuracy, six phrase/control sequences, 36 native/ONNX cases, exact non-lid coefficients, unchanged lip gaps and protected files. Lid-only 4–15 Hz RMS falls 88–96%, and all six held-out coefficient errors improve. Partial blink labels do not establish complete spontaneous blinks. James now serves eyelid-stable-v1, SHA 7ffcd937b581, 2,933,618 bytes, in visual-review mode retaining the known +5.5% anger lip-gap result. Player and 2.5D remain untouched. Set A2F_SOURCE_APP to the task-local .local/research-source-snapshot for the SHA-matching baseline/rig when the historical source cache is unavailable. Browser checked all three phrases at neutral and anger, then returned to neutral appointment.


Same-audio teacher comparison added for Maeve neutral “Emma moved my camera from the middle shelf.” Raw teacher 88831-output geometry is replayed directly on the left; candidate remains on the right; released v4 remains selectable. Frame count, sample grid, audio SHA, model SHA and official neutral/rest identity checked. At the most mobile teacher landmark, swing is 2.37 mm raw, 2.14 mm fitted teacher, 1.51 mm student. Plot includes all five landmarks. Browser playback/seeking synchronize correctly; non-neutral controls disable the neutral replay. Both builds pass, 53 protected files unchanged. No model training, Player or 2.5D changes.


Mouth-tail follow-up: `diagnose_mouth_tails.py` compares all five clips at neutral
and anger through raw teacher, fitted teacher, v4 and current candidate. Extra
corner motion and half-second join artifacts were confirmed; appending silence
leaves original frames unchanged. `refine_mouth_tails.py` trains the fast branch
for teacher corner positions/velocities and overlap consistency, with extra tail
weight and raw lip-gap guards. The unguarded pass failed direct lip-gap checks.
`guard_mouth_tails.py` adds 128 bounded affine/box projections to preserve the
parent's nine lip gaps approximately while allowing corner changes. The guarded
model improves all five neutral corner tails (21% aggregate), all-case corner
error 16%, join-step error 21%, and F5 neutral about 30%. Lip-gap RMS changes at
most .003 mm in the ten checks; maximum individual gap change is .098 mm.
Brow/lid/gaze coefficients are unchanged. Sixty native/ONNX cases pass.
The guarded model still fails the predeclared corner-regression check for
strong-anger appointment (0.91 to 1.09 mm error). It is staged only for explicit
research visual review, with that warning and the existing +5.5% lip-gap result
visible; it is not a fully passing/promoted model. Model SHA 25fcbfd81963,
3,519,076 bytes. Player and 2.5D remain unchanged. Browser checked all five neutral
phrases and the full-anger appointment; F5 is left selected against raw teacher.
Teacher references are now per-recording, with matching clips/controls enforced.
See outputs/mouth_tail_review_results.md for measurements and limitations.


## Strong-anger appointment corner correction

The regression now passes: raw-teacher ending-corner RMS error is 0.588 mm,
versus 1.095 mm in the previous review and 0.911 mm before the first corner
refinement. V4 is smoother, but its equivalent anger comparison has 1.356 mm
corner error because it has no matching anger control. The new corner correction,
not the released v4 path, introduced the extra movement.

The final implementation fits the fast output projection through a bounded
lip-gap constraint, using appointment/anger as explicit development regression
training alongside the voice corpus. The deployed correction uses only 27 jaw
and mouth coefficients; the other 25 controls (including cheeks, nose, brows,
eyelids and gaze) retain the previous stable tracks. This matters because broad
cheek corrections could move the eyelid mesh even with fixed eyelid coefficients.
Full-surface jitter was rechecked, rather than relying on coefficient identity.
The lip-gap guard is approximate and uses canonical James geometry; it is not a
new 2.5D calibration or a production-ready universal avatar constraint.

All ten phrase/control corner checks, retention of neutral gains, lip-gap checks
and 60 native/ONNX parity cases pass. Appointment/anger lip-gap RMS stays 3.269 mm;
maximum change versus previous review across ten cases is 0.000221 mm. Brow-region
fast motion changes by at most 0.01%, eyelid-region by 4.74%. A 627-window held-out
voice check across all eleven conditions confirms exact identity of 25 non-mouth
coefficients and lip-gap RMS ratios within 0.0006% above baseline. The pre-existing
+5.5% lip-gap comparison remains explicitly labelled; it is not the corner bug.

The exact regression phrase was used for development training and selection, so
it is not independent validation. The other phrases and held-out voices were
rechecked. All five phrases were played in the browser at neutral and anger.
James is left on appointment, anger 100%, raw teacher left and corrected model
right; both reach frame 307 together. Player, released models, v1 textures and
2.5D mapping remain unchanged. Sixty parity cases and protected-file checks pass.

Current review: mouth-corner-isolated-v1, SHA 23fc6ec7afe2, 3,807,579 bytes.
Sources: prepare_corner_regression.py, refine_projected_corners.py,
refine_projected_corners_stress.py, mouth_corner_model.py,
export_mouth_corner_model.py and check_corner_refinement.py. The older general
and broader-face variants were not promoted after failing case/upper-face checks.
Run prepare_teacher_review.py --sample appointment --emotion anger to add the
matched raw-teacher replay; references are indexed by recording and exact emotion.
Partial strengths without a matching replay fall back to the v4 comparison.
Outputs: anger_corner_fixed_results.md, anger_corner_fixed_verification.json,
anger_corner_dev_integrity.json, anger_corner_fixed_comparison.png and the
James review screenshot in the task outputs directory.

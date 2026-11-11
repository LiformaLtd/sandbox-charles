# Expressive ARKit research

## Combined 52-channel candidate and James browser review

The face pilot is stored in `.local/expressive-arkit-v2`. A separate gaze branch
uses all three official NVIDIA identities and is stored in
`.local/expressive-arkit-v3` (these are experiment directories, not released
speech-model version numbers). It conditions gaze on explicit Claire/James/Mark
reference styles instead of averaging their eye directions. No DAWN labels are
used. Head rotation remains deferred.

After the face pilot below, reproduce the combined candidate with:

```sh
python scripts/research/expressive_arkit/gaze.py labels
python scripts/research/expressive_arkit/gaze.py features
python scripts/research/expressive_arkit/gaze.py train
python scripts/research/expressive_arkit/export_full.py
python scripts/research/expressive_arkit/combined_metrics.py
```

`A2F_FULLFACE_OUTPUT` optionally overrides the combined candidate's output
directory. The final `fullface.onnx` includes both `blendshapes` and the unchanged
`articulation` stream. The latter is a reference for future contact composition;
it does not currently repair the expressive mouth. Strong-expression closure and
speech-time tongue-proxy problems are visible in the first 3D review.

Run the isolated James comparison:

```sh
npm ci
python scripts/research/expressive_arkit/prepare_review.py
npm run dev:expressive
```

Open the URL Vite prints, normally `http://127.0.0.1:5173/expressive.html`.
The research configuration redirects its root there and uses the next free port
if 5173 is occupied. `npm run build:expressive` produces the research build.
The UI compares v4 and the candidate using the same audio clock, supports uploads
and a manual rig-target inspector, and fixes head motion. This CPU view does not
change the regular demo entry point, model selection, head worker or Player.

For portable offline inference on the combined graph:

```sh
python scripts/research/expressive_arkit/infer_full.py \
  --model /absolute/path/to/fullface.onnx --audio /absolute/path/to/mono-16khz.wav \
  --emotion anger --gaze-style Claire --output /absolute/path/to/animation.npz
```

Run all contract tests with the explicit research environment set:

```sh
python -m unittest discover -s scripts/research/expressive_arkit -p 'tests*.py' -v
```

See `docs/plans/expressive-arkit-pilot.md` for the actual checks, failures and gates.

## First face-adapter pilot

This is an isolated teacher-label and small-model pilot. It changes no Player,
2.5D map, published model, browser runner or existing demo entry point.

The frozen finetune-r4 speech predictor supplies audio features and articulation.
A new 64-channel temporal adapter accepts those features and ten explicit NVIDIA
emotion controls and produces 52 bounded coefficients. At neutral, all previously
learned coefficients and the four newly enabled lateral mouth/jaw coefficients
are protected. Emotion-conditioned differences come from matched teacher runs
with the same recording, identity, noise and recurrent warm-up.

The labels retain 43 skin-solved coefficients, eight derived gaze controls, a
tongue protrusion proxy, the original four eye rotations, sixteen tongue controls,
fifteen jaw values, and per-frame skin-fit residuals. Large raw geometry tracks
are processed in memory rather than duplicated on disk.

Important distinctions:

- Emotion is supplied explicitly. This model does not infer a requested emotion
  from neutral TTS or independently classify the audio's emotional state.
- Gaze has a declared research endpoint of 25 local degrees. Its signs, scale,
  eyelid coupling and per-avatar endpoints need visual validation.
- The supplied skin `tongueOut` target is empty. This experiment uses the real
  `tongueStretch` shape as a documented protrusion proxy after subtracting
  authoring jaw/mouth motion. It is not an ARKit capture calibration.
- Head orientation is separate from ARKit's 52 facial channels. The released
  NVIDIA Multi v3.2 export lacks the paper's experimental head extension.
  Head training is deferred; DAWN is not used for labels.
- A dense 52-output graph does not prove full range or semantic quality. The
  coverage report must be read alongside accuracy and closure checks.

## Reproduce

Use the existing Mac research environment: Python 3.12, NumPy, SciPy, Torch 2.4.1,
ONNX, ONNX Runtime, MLX/MLX Metal and Matplotlib. The existing pinned teacher
runner is a deliberate read-only external dependency. No new service is started.

Set explicit paths (there are no downloads or hidden asset substitutions):

```sh
export A2F_RESEARCH_WORK=/absolute/path/to/historical/work
export A2F_TEACHER_ROOT=/absolute/path/to/liforma-audio2face-v3/ai-workflows/apple-silicon
export A2F_SOURCE_APP=/absolute/path/to/sandbox-feature-distill/sta/audio2face
export A2F_ASSET_ROOT=/absolute/path/to/v3-model
# Optional; default .local/expressive-arkit-v2 inside this app.
export A2F_EXPRESSIVE_OUTPUT=/absolute/path/to/new/output

python scripts/research/expressive_arkit/tests.py
python scripts/research/expressive_arkit/audit.py
python scripts/research/expressive_arkit/prepare.py labels
python scripts/research/expressive_arkit/prepare.py features
python scripts/research/expressive_arkit/train.py
```

`prepare.py` freezes the label recipe and manifests and verifies source audio
hashes. Use a new output directory when changing a recipe. A completed training
trajectory is not overwritten. The existing recordings are held out by prompt
and voice within this pilot; they are not fresh independent release confirmation.

`report.py` additionally expects the existing appointment regression WAV at its
recorded local path and a `protected-before.json` path/hash snapshot. It compares
with a constant expression-pose control, checks all emotion conditions on the
regression, and verifies protected files. These are explicit local review inputs,
not dependencies of the model or offline runner.

The first attempt's fixed ridge over-damped weak eyelid channels in synthetic
self-recovery. Its partial labels were stopped and retained under the v1 output
with the rejected fitter. V2 uses column-energy-relative regularization and
passes a separate synthetic recovery check before label generation.

## Run a candidate

```sh
python scripts/research/expressive_arkit/infer.py \
  --model /absolute/path/to/model.onnx \
  --audio /absolute/path/to/mono-16khz.wav \
  --emotion amazement --intensity 0.6 \
  --output /absolute/path/to/animation.npz
```

The exported graph accepts `pcm [1,16000]` and `emotion [1,60,10]`, and returns
`blendshapes [1,60,52]`. The runner retains the middle 30 frames of overlapping
one-second windows, aligned to the existing 60 fps sample grid. Neutral controls
are all zero. Emotion order is recorded in the label recipe. Transitions and
intermediate intensities require additional validation; the pilot trains neutral
and the ten one-hot, full-intensity conditions.

This graph adds an adapter and an emotion input. It is **not** compatible with the
existing custom WebGPU weight loader without an explicit runner extension.
Native/ONNX CPU parity is separate from browser and mobile performance testing.

## Full neutral performance and stable brows

The revised experiment uses absolute teacher labels at neutral for every learned
face channel. The frozen compact speech core supplies features and an
articulation prior; no inference gate forces the active neutral face back to v4.
The mouth coefficients may change. The selection guard is macro lip-gap RMS
against raw teacher geometry, with at most 5% regression on each neutral
validation cohort. Direct phrase checks additionally apply that tolerance to each
of the three James clips at neutral and full anger. This supersedes the earlier
requirement to preserve mouth coefficients exactly.

`full_neutral_data.py` reruns the official teacher for absolute facial targets and
nine raw lip gaps, refits brows regionally, and gates the tongue-out proxy using
actual protrusion past the moving lip boundary. `full_neutral_model.py` separates
a six-anchor-per-second brow decoder from the fast face correction. Every learned
channel uses neutral teacher supervision; eight gaze channels retain the existing
teacher-trained gaze head. Head motion remains deferred.

Set `A2F_NEUTRAL_OUTPUT` to a fresh absolute output directory, then run:

```sh
python scripts/research/expressive_arkit/full_neutral_data.py labels
python scripts/research/expressive_arkit/full_neutral_data.py features
python scripts/research/expressive_arkit/full_neutral_data.py replay
python scripts/research/expressive_arkit/train_full_neutral.py
python scripts/research/expressive_arkit/tests_full_neutral.py
python scripts/research/expressive_arkit/verify_full_neutral.py
python scripts/research/expressive_arkit/prepare_review.py \
  --verified-candidate "$A2F_NEUTRAL_OUTPUT"
```

Staging requires a passed verification report whose SHA matches the model.
Content-addressed model/reference filenames and a manifest written last prevent
mixed model versions during browser reloads. The browser checks active lip-gap
RMS using raw teacher reference tracks for the measured clips and controls.
Uploaded audio and unmeasured conditions are explicitly reported as unmeasured.
The separate frozen articulation output checks core integrity; it is not the
criterion for the active mouth.

The first full-neutral pass passed the broader validation lip gates but failed
appointment phrase gates (+5.7% neutral, +13.1% anger). It was retained as a
rejected experiment, not staged. The second pass increases raw gap supervision
and historical mouth replay. These known regression clips are development
checks; they are not fresh independent release confirmation. Exact dense teacher
reproduction, all internal tongue modes and head motion are not provided by this
52-channel approximation. Player and 2.5D mapping remain unchanged.


Final refinement outcome: all three neutral phrase RMS checks pass (+2.99%, -9.16%, -16.03%). Appointment at full anger remains +5.47%, narrowly above the 5% working tolerance, so the candidate was not staged. Strong-anger brow jitter fell 90.9–94.7%, but some slow neutral brow gestures remain too weak. A discarded run with an incorrect loop length was rerun using the correct 1,200-step schedule. Player and 2.5D mapping remain unchanged.

For explicitly requested visual review of a candidate with known failed gates,
use `prepare_review.py --review-candidate /absolute/path/to/candidate-directory`.
This still checks the model SHA and displays the failed measurements; it allows
playback above the lip-gap threshold in this research view only. The strict
`--verified-candidate` mode continues rejecting failed candidates. Per the user's
request, the James view now serves `expressive-full-neutral-v3` (SHA 631b4d010f2f),
including the known +5.5% appointment/anger result. Player and 2.5D are unchanged.


Eyelid follow-up: confirmed excess squint/widening/left-blink oscillation in the model. `train_eyelids.py` fits separate six-Hz squint/wide and twelve-Hz blink decoders against absolute teacher labels; parent mouth/brow/gaze/other channels are frozen. `diagnose_eyelids.py` compares raw teacher, fitted teacher and model geometry; `verify_eyelids.py` checks held-out lid accuracy, six phrase/control sequences, 36 native/ONNX cases, exact non-lid coefficients, unchanged lip gaps and protected files. Lid-only 4–15 Hz RMS falls 88–96%, and all six held-out coefficient errors improve. Partial blink labels do not establish complete spontaneous blinks. James now serves eyelid-stable-v1, SHA 7ffcd937b581, 2,933,618 bytes, in visual-review mode retaining the known +5.5% anger lip-gap result. Player and 2.5D remain untouched. Set A2F_SOURCE_APP to the task-local .local/research-source-snapshot for the SHA-matching baseline/rig when the historical source cache is unavailable. Browser checked all three phrases at neutral and anger, then returned to neutral appointment.


Neutral brow demonstration: stage the current review candidate with
`--extra-sample train-phonon-maeve-001 --extra-sample test-phonon-hugh-001`.
Maeve says “Emma moved my camera from the middle shelf.” and gives a measured
1.62 mm 5th-to-95th percentile swing at the most responsive canonical brow
landmark with all emotion inputs zero. This is a known training recording,
selected for demonstration, not independent validation. Hugh’s held-out
“Come home before the warm evening meal is ready.” gives 0.91 mm. Delivery and
voice matter; the words alone do not guarantee the same motion. These two added
review clips have no staged raw lip-gap references, so that metric is explicitly
unmeasured in the browser. The model is unchanged.


Raw teacher comparison for the neutral Maeve example: after staging the review
assets and extra samples above, run `prepare_teacher_review.py`. It reruns
official NVIDIA Multi v3.2, neutral James, seed 0, on the identical SHA-checked
recording; validates the 60 fps sample grid and renderer neutral mesh; and stages
a hash-checked dense replay. The browser defaults to raw teacher versus candidate
for this recording, synchronized to the same audio clock. Released v4 remains a
comparison option. Unsupported recordings/controls use v4 rather than a mismatched
teacher replay. No network teacher service or model change is involved.
The optional `--sample` flag selects another staged recording with cached fitted
teacher labels. Restaging `prepare_review.py` preserves existing teacher replays when the staged audio and replay hashes match. Run `prepare_teacher_review.py --sample STEM` to add further references.
The local report includes raw geometry, fitted-teacher and candidate landmark
traces, with mean-centering explicitly labelled. No artificial motion gain.


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

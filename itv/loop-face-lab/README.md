# Loop Face Lab

A standalone SvelteKit 3 experiment: existing video loops provide body motion and
head pose, a neutral source image provides the face texture, and browser-generated
speech drives a replacement MediaPipe mesh. No video generation is exposed.

## Run

Node 22.18+ and Chrome with WebCodecs, WebGL, WebAssembly SIMD and optionally
WebGPU are required. The app has its own dependencies and generated directories.

```sh
npm ci
npm run assets -- /absolute/path/to/audio2face-2d/static/assets /absolute/path/to/face_landmarker.task /absolute/path/to/phonon-31m-en
npm run dev
```

The server tries **127.0.0.1:5173**, then the next free port. Read Vite's printed
URL. It never requires a 30xx frontend port. The optional saved-loop library is
read through the existing LTX-2.5 server at `127.0.0.1:3026`. `.env.example`
documents `LTX25_API_URL`. Source image/video uploads, tracking, speech and
playback work without that service. Only GET routes are proxied; generation and
retry requests are not implemented.

The third input is the already-downloaded Phonon bundle from the local
`ai-workflows/apple-silicon/phonon` setup. Its pinned manifest verifies each selected
voice file before copying; serving never downloads voices from external hosts.
For an existing installation, add/update just the voices with
`node scripts/prepare-voices.mjs /absolute/path/to/phonon-31m-en`.

The asset command reads the existing Face Lab's already-prepared files, without
modifying that app. It copies Phonon's Q8 runtime assets and eight catalogue voices, the verified
2,251,827-byte compact model (SHA-256
`5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd`),
contact calibration, one calibrated mapping template, and a local MediaPipe model.
MediaPipe's matching npm WASM files are copied too. Their Emscripten module-worker
adapter exposes `ModuleFactory`; the worker supplies the expected dynamic-import
and debug hooks. No runtime models are fetched from third-party hosts. Asset
provenance is written into the ignored asset directory. Do not commit models,
private images, audio, videos, dependencies or generated build files.

This read-only preparation depends on another app in this repository
(`sta/audio2face-2d`). The optional library depends on the local HTTP API in the
parent workspace's `ai-workflows/apple-silicon/ltx25`. There are no implicit
filesystem imports or writes to either dependency at runtime.

## Use

1. Load a **Loop ZIP** exported from Loop Lab to restore the source image, both motion groups and durations together. Alternatively choose a saved Loop Lab set, select one or more ready clips, and press
   **Load selected loops & source image**. Alternatively upload the original
   image and one or more videos, then **Prepare face tracking**.
2. Every decoded presentation frame is processed by MediaPipe once, in a worker.
   The app records 478 landmarks and the facial transformation matrix. It caches
   complete tracks by video content hash plus tracking revision in IndexedDB.
   Repeated playback and refresh reuse them. Failed detections stay explicit;
   those frames show the original video and a notice.
3. **Start loops** plays up to ten seconds per selected clip, choosing a different
   clip at each join when more than one is present. Two preloaded decoders and a
   persistent canvas preserve the existing Loop Lab join behavior.
4. Choose a **Phonon voice**, enter English text, then **Generate & speak**. Phonon runs in a browser worker;
   the compact model converts the exact playback PCM into 52 ARKit coefficients
   at 60 Hz. WebGPU is the default; CPU/WASM can be explicitly selected or set
   with `?runtime=cpu`. There is no silent runtime substitution.
5. The existing ARKit-to-MediaPipe sampler, contact correction and calibrated
   linear displacement field are reused. A local affine transfer adapts the
   template's displacements to the newly fitted source geometry. The original
   video's mouth expressions are excluded from the replacement. Rigid anchors
   set translation/scale, and MediaPipe rotations set roll, pitch and yaw.
6. Speech follows its audio clock independently of the random video joins. When
   speech ends, the replacement returns to its source expression until the talking
   clip finishes, then switches to original idle videos when available. **Stop** pauses loops and speech; Start resumes them. Audio controls
   can seek/replay the speech. Original loop audio remains muted.
7. **Neutral face & neck** is enabled by default. It renders a neutral source-image
   face and an eight-row neck extension below the speaking face. The inner neck
   rows stay opaque to hide the original lowered jaw. Head motion reduces toward
   a smoothed image-plane anchor; alpha fades separately at the outer edge.
   The largest tracked jaw drop expands the minimum coverage once at preparation,
   keeping the mask stable during playback. **Neck coverage** can extend it further.
   **Neutral layer only** isolates this underlay. **Show mesh** draws its grid in
   amber and the speaking mesh in green. Coverage and enablement survive refresh.
8. Compare **Replace face** on/off, **Show mesh**, and motion strength. Pose values
   are shown in the viewer. The image, selected files and speech text are saved
   in IndexedDB; a refresh restores the scene and cached tracking, but waits for
   Start/Generate before playback. Speech audio itself is regenerated on request.

IndexedDB belongs to an origin (scheme + host + port). Returning on the same port
restores local uploads. Saved Loop Lab sets can be imported from another port
through the local library, without generating videos again.

## Idle and talking motion

Use **Add or load videos as** to choose **Idle motion** or **Talking motion** before
uploading files or loading a saved set. Loading a set replaces that group while
retaining the other group. Both groups must use the same original source image.
Each clip also has a motion-type picker. Existing saved clips migrate to Talking.
Group assignments and both video sets persist in IndexedDB.

With both groups loaded, Start randomly loops idle clips using the original video
without the replacement face or neck layer. Preparing speech leaves idle playback
running. Once the audio is ready, a talking clip starts from its beginning before
audio playback. While speech continues, talking clips cycle randomly. Audio ending
or being paused immediately restores the neutral source expression; that talking
clip finishes before playback returns to idle. Replaying audio interrupts idle in
the same way. Stop pauses both video and speech. Speech mute changes only audio
volume and does not change the motion state. Talking-only legacy scenes still play
with a neutral replacement while silent; add idle clips for the full cycle.

## Browser voices

The **Phonon voice** picker has female and male groups, with UK/US accents:

- Female: Zoey, Juniper, Freya and Maeve.
- Male: Easton, Trevor, Alfie and Hugh.

Gender/accent labels and source voice IDs come from Gradium's pinned Phonon voice
catalogue revision `0ee17205f254b5abc43b266c8a0f215565c352f7`.
The original Easton voice remains the default. Selection is saved immediately to
IndexedDB (`speech-voice`) and restored on refresh before generation. It is scoped
to this app's origin, like the uploaded scene. Changing voices clears old speech
and facial animation while leaving the loop playback available.

All eight small voice embeddings are served locally. The existing Phonon engine
and Q8 model stay loaded while the page is open; only newly used voice embeddings
are loaded when switching. No server-side TTS is used. The generated recording
shows the voice actually used. Speech samples for that voice feed the same
Audio2Face inference/playback pipeline. Source voice files and their checksums
are recorded in the ignored `lf-tts/voice-provenance.json` asset manifest.

## Fixed-size 3D fitting

**Fixed 3D face size** is enabled by default. The source image defines one fixed
3D reference mesh. Eye corners (33/133/362/263), nose (168/6/1/4) and temple-side
points (127/356) fit an orthographic camera: one shared scale for all loaded
clips, a proper 3D rotation and an image-plane translation for each frame. The
face mesh has no actual ear landmarks; the temple-side vertices are used instead.

Scale is calibrated once from robust size measurements over all selected frames,
normalized by the shorter frame dimension so different resolutions use the same
camera framing. Each pose is then refitted at that locked scale with Huber
weighting and a weak orientation prior to the measured MediaPipe rotation.
Rotations remain orthonormal with positive determinant; scale, shear, and
independent landmark deformation are not per-frame degrees of freedom.

Speech keeps the selected anchor vertices at their reference 3D positions; the
mouth, eyelids and remaining expression geometry can still animate. Both the
speaking face and neutral face/neck layers use the same fitted pose. Raw landmarks
are retained for chin coverage and comparison. Nothing retrains or changes the
speech model. Fitting runs once during preparation from cached MediaPipe outputs,
then playback reuses the poses. No detector re-run is needed on refresh.

Toggle the control to compare the previous per-frame sizing. **Show fitted
points** draws detected points in coral and projected rigid anchors in mint,
with lines showing residual error. The panel reports detected scale variation and
median reprojection error. The size-lock choice persists in IndexedDB. Nonzero
error is expected when the generated face changes shape or violates the camera
model. A locked camera scale deliberately cannot follow real zoom/depth-induced
size changes. The fixed-size fit provides geometric regularisation; the optional
rotation filters below handle temporal fit noise.

## Yaw and pitch smoothing

**Smooth yaw** and **Smooth pitch** are enabled by default with independent
300 ms timescales (100–800 ms controls). They operate on the prepared fixed-size
pose track. **Roll is never smoothed.** Translation and camera scale also retain
exactly their fitted values. Disable either filter for a direct comparison. The
filters are active when **Fixed 3D face size** is enabled; all choices persist.

Each enabled angle uses symmetric, time-aware local quadratic regression with
Gaussian weights and robust outlier reweighting. Because complete clips are
available beforehand, it can use both earlier and later samples without adding
causal playback delay. Slow turns retain their trajectory; stronger settings can
soften genuine fast turns as well as noise. Angles are unwrapped before filtering,
and missing detections remain missing. Long detection gaps split the smoothing
neighbourhood. The original fitted pose track stays available for comparison.

These loops deliberately share their source-image endpoints. Each filtered angle
therefore eases to a robust common endpoint value over at most 350 ms at each
end, with zero endpoint slope. This protects same-image loop joins; it assumes the
loaded clips genuinely return to that source pose. Roll/position at joins remain
as measured. This is not general motion interpolation for arbitrary unrelated
video cuts.

Rotations are reconstructed from filtered yaw/pitch plus the original roll, so
the mesh stays a proper rigid rotation at fixed scale. The neutral face/neck and
speaking surface share the resulting pose. Landmark fit errors are recomputed
after filtering. Reported angular roughness is RMS angular acceleration away from
the endpoint fades; it measures trajectory roughness, not perceptual quality.
Filtering reuses cached tracking and the original fitted poses, with no new face
inference, speech inference or video generation.

## Scope and visual limits

This is a research compositor. The displacement transfer is an adaptation of an
existing avatar-calibrated mapping, not a fresh multi-viseme calibration for each
new image. It is not a production-quality face replacement model.

The renderer warps one source texture over a rigidly posed, animated mesh and
feathers the boundary into the video. The neutral neck anchor uses a symmetric
smoothed estimate of head translation/scale, with reduced movement and the source
rotation, rather than an independently tracked torso. This keeps the lower neck
from rigidly following head turns, but cannot guarantee alignment during independent
body motion. A tightly cropped source image limits the available neck texture and
produces an explicit notice. The layer can overpaint hair, collars or hands.
A flat mouth cavity has no synthesized teeth or tongue. One image cannot supply newly visible surfaces during large turns or
changing lighting. Hair, hands or other occluders crossing the face are not
segmented. These limits can reveal seams or overpaint occluders. Use compatible,
mostly frontal clips with a clearly visible face and the matching neutral image.

## Code reuse

Copied, scoped browser modules from `sta/audio2face-2d/src/lib`:
`speech`, the Phonon worker, `inference` and its worker, `windows`, `audio`,
`shape`, `contact`, `types`, and `vendor/compact-*`.

`player.ts` derives from the existing LTX-2.5 Loop Lab dual-deck player, with a
per-frame compositor hook and support for shorter clips. Pose extraction,
content-addressed frame caching, arbitrary-image retargeting, the transparent
face renderer, and this UI are specific to this experiment. Existing apps and
production animation paths remain separate.

## Check

```sh
npm test
npm run check
npm run build
# Against the running app and an existing generated character set:
TEST_BASE_URL=http://127.0.0.1:5175 npm run test:browser
```

Geometry tests verify neutral projection on non-square images, rigid translation,
separation from the video's mouth expressions, scaled pose matrices and timestamp
selection including missing frames. Neutral-layer checks cover source UV alignment,
opaque coverage of a lowered chin, separate motion/opacity falloffs, head rotation
isolation at the lower neck and explicit limits for cropped source images.
Rigid-fit tests cover injected scale drift, fixed 3D anchor distances under
rotation, biased pose recovery, outlier rejection, mouth isolation, speech anchor
pinning, shared calibration across resolutions, and missing frames. Browser integration tests read existing
character media, verify all 241 frames per clip, real Phonon + compact CPU and WebGPU
inference, loop switching, audio-driven animation, cached reload, file uploads,
rejection of generation routes, paused layer comparison and persistent neck controls.
Voice integration checks synthesize real female/male speech, verify distinct WAV
outputs, reuse of the loaded model, clearing old speech and IndexedDB restoration.
The chin regression uses the original talking clip's worst jaw-drop frame, verifying
that the old layer is transparent at the protruding chin while the new one is opaque. They make no model-generation jobs.

Development/preview send cross-origin isolation headers. A standalone adapter-node
production server must preserve the same COOP/COEP headers when hosted behind a
proxy. This experiment is configured for local use; no deployment is configured.

## Speech duration scheduling

Before audio starts, the complete recording gives its exact duration. A portable
11-clip bank offers talking lengths 3–9 seconds. For up to 9 seconds, use the
shortest available clip that covers the rounded-up duration (6.4s → 7s). For longer
speech, use the fewest clips that cover it, balanced to within one second
(12.4s → 6s + 7s; 18.1s → 6s + 6s + 7s). The planned sequence is shown beside
the recording. Repeated lengths can reuse a clip via the two independent decks.
Idle clips are excluded from the speech plan.

After speech ends, finish the final talking clip with a neutral face, then return
to random original idle video. Stop/Start resumes the current clip; replaying a
finished recording restarts its planned sequence. Audio and speech animation keep
the same continuous audio clock across all video joins. Older uploads infer a
duration from decoded video; their available lengths are used rather than
pretending they contain the new 3–9 second bank.

The versioned ZIP manifest is validated before replacing the existing scene.
Group assignments, durations, source image and videos survive refresh in IndexedDB.
Archives are processed locally with `fflate`; no files are sent to third parties.

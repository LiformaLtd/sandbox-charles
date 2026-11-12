# MediaPipe mapping feasibility probe

Standalone research on actual v1 avatar viseme meshes. This does not import,
modify or start the player. No service, port, credentials or deployment is used.
The current candidate **fails** its quality screen; do not ship its map files.

## Inputs and ownership

Pass all input locations explicitly. They are read only:

- `--avatars`: the existing Liforma `source/avatars` directory, containing
  `index.json`, each face's `face-meshes.json`, and original viseme images.
- `--sandbox-assets`: the audio2face sandbox's existing `.local` directory,
  including the frozen `compact-v4-validation/v4-candidate.onnx`, source rig
  `rig.json`/`rig.bin`, and `compact-v4-confirmation` PCM/prediction caches.
- `--source-rig`: the matching James source bundle with `bs_skin_James.npz`
  and its configuration. The pinned model SHA is checked by the script.
- `--face-task`: the existing `face_landmarker.task` used by CDN authoring.
- `--out`: a new ignored research result directory. Never point it at masters.

This deliberately reads avatar-authoring assets and source-rig research assets
across repositories. Those explicit input paths are the only cross-repository
dependencies; production apps have no dependency on this experiment. Model
weights, photos, generated registration/basis files and research output stay out
of Git. Do not publish or migrate avatar assets as part of running the probe.

## Geometry matching and noisy observations

The latest experiments use actual geometric source-pose matching, corrected
source lip-curve correspondence, optional extended calibration poses and robust
observations. They are research candidates, **not cleared for player integration**.
An old fitted mesh is not assumed to be exact ground truth. Clear contact defects
and isolated point errors can be repaired in calibration copies, while reference
identity, reliable expression variation and original texture UV data are preserved.

Use a new output directory for each frozen protocol. For the robust linear probe,
add `--assignment geometry --registration curves --calibration-set extended --robust`
to the `run_probe.py` command below. Four development faces establish the source
assignments; the other faces do not establish their own test coefficient queries.
Per-face absent auxiliary examples are explicitly recorded. Test the result with:

```sh
.venv/bin/python sta/audio2face/scripts/research/mediapipe_mapping/noise_check.py \
  --run /absolute/path/to/robust-run \
  --avatars /absolute/path/to/source/avatars \
  --sandbox-assets /absolute/path/to/audio2face/.local

.venv/bin/python sta/audio2face/scripts/research/mediapipe_mapping/safety_check.py \
  --run /absolute/path/to/robust-run \
  --avatars /absolute/path/to/source/avatars \
  --sandbox-assets /absolute/path/to/audio2face/.local

.venv/bin/python sta/audio2face/scripts/research/mediapipe_mapping/run_local_probe.py \
  --input /absolute/path/to/robust-run \
  --avatars /absolute/path/to/source/avatars \
  --sandbox-assets /absolute/path/to/audio2face/.local \
  --out /absolute/path/to/new-local-residual-run
```

The local residual runner tests an anatomical prior and Gaussian correction. Its
parameter search uses development faces only. It is a separate candidate and did
not pass withheld-pose quality. Input-noise tests measure both average error and
maximum point displacement: an average pass must not conceal one collapsing lip
vertex. Triangle orientation in the image plane is a conservative diagnostic,
not definitive proof of a rendering defect; compare the existing asset baseline.
Coherent but wrong full meshes or a bad identity reference require image evidence
or additional trusted observations rather than an unverified geometric guess.

## Run

From the dedicated sandbox checkout:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  -r sta/audio2face/scripts/research/mediapipe_mapping/requirements.txt

.venv/bin/python -m unittest discover \
  -s sta/audio2face/scripts/research/mediapipe_mapping -p 'test_*.py' -v

.venv/bin/python sta/audio2face/scripts/research/mediapipe_mapping/run_probe.py \
  --avatars /absolute/path/to/source/avatars \
  --sandbox-assets /absolute/path/to/audio2face/.local \
  --source-rig /absolute/path/to/v3-model \
  --face-task /absolute/path/to/face_landmarker.task \
  --out sta/audio2face/.local/mediapipe-mapping-pilot-v3

.venv/bin/python sta/audio2face/scripts/research/mediapipe_mapping/make_report.py \
  --run sta/audio2face/.local/mediapipe-mapping-pilot-v3 \
  --output /absolute/path/to/deliverables
```

The input manifest records hashes of actual fitted meshes, source bundle, model
and task. One cached coefficient sequence is checked against a fresh exact ONNX
replay before cached predictions are reused. `protocol.json` freezes numerical
limits before the face fit runs. The regularization sweep is separately labelled
exploratory; it does not change the primary thresholds.

## Results and limits

Read `mapping-feasibility-results.md`, `results.json`, the raw viseme inventory,
source-library definitions, and regularization sensitivity output together.
Source and target landmarks are represented in separate coordinate frames;
comparison overlays transform cohort destination geometry back to original
image space using stable upper-face anchors. They do not modify or warp images.

The automatic source registration and source-example assignments are preliminary.
The lip-contact check is a bounded image-plane constraint, not production oral
compositing or physical lip simulation. Rest-like frames are tested separately
from active source closure so naturally parted target rests are not falsely
required to close. A good training reconstruction or forced lip seal does not
establish generalization. Check held-out visemes, correction magnitudes and
source-pose semantics before moving on to any player work.

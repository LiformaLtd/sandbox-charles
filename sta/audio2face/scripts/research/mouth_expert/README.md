# Frozen mouth-expert compatibility

[Protocol](../../../docs/plans/mouth-expert-compatibility.md) ·
[Results](../../../docs/plans/mouth-expert-compatibility-results.md)

No model training or demo deployment. This screens the frozen paper-distributed
Auto-AVSR expert and secondary SyncNet on teacher-fitted James corruptions.
A post-screen independent confirmation tests the selected SyncNet without tuning.

Prerequisites: current app rig/packed blendshapes, previous compact-loss and
compact-feature confirmation corpora (audio, manifests, phones, fitted teacher
labels), and `arkit-student` rig metrics/channels under `A2F_RESEARCH_WORK`.
Set that variable to the existing research asset workspace. Immutable previous
artifact directories can be symlinked into this dedicated worktree's `.local`.
Run `npm ci` here; never share writable node_modules.

All new artifacts are `.local/mouth-expert-v1`. Under its `vendor/`, clone:

- https://github.com/kaist-ami/3d-talking-head-av-guidance at
  `7e4c63bb4f9da8d3d932d19c89bc1c34d25cb70e`.
- https://github.com/mpc001/auto_avsr at
  `8acf580bcab10507532854ca5371f76ca1c364a5`.
- https://github.com/joonson/syncnet_python at
  `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623`.

Download the official AV-guidance pretrained lip-reader into `vendor/lipreader.ckpt`
from its README's Google Drive file ID `1mU6MHzXMiq1m6GI-8gqT2zc2bdStuBXu`.
Download `syncnet_v2.model` and optional `example.avi` from
`https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/`, naming the example
`vendor/sync-example.avi`. Verify checkpoint hashes in the results/protocol.
These large third-party files and corpora must not enter Git. Consult their
original repositories for provenance/terms.

CPU environment: Python3.12, torch2.4.1, NumPy2.2.6, OpenCV4.12.0, SciPy1.16.3,
PyYAML, SentencePiece and `python_speech_features==0.6`. The last package may be
installed with `pip --target .local/mouth-expert-v1/vendor-deps --no-deps ...`.
Direct E2E imports avoid Lightning/Hydra/CUDA training dependencies. We do not
modify vendor model implementations or adapt their weights. Use `TRAIN_PYTHON`
for this environment. Node22+ /Playwright1.65.0 /WebKit2388 render the real rig.

From the app directory with fresh outputs:

```sh
"$TRAIN_PYTHON" scripts/research/mouth_expert/prepare.py
# Start npm run dev -- --mode hosted and use its actual next-free port.
export DIAGNOSTIC_URL=http://127.0.0.1:ACTUAL_PORT
export PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs
export WEBKIT_EXECUTABLE=/absolute/path/to/webkit/pw_run.sh
export A2F_EXPERT_OUTPUT="$PWD/.local/mouth-expert-v1"
"$TRAIN_PYTHON" scripts/research/mouth_expert/freeze.py
node scripts/research/mouth_expert/render.mjs
mkdir -p "$A2F_EXPERT_OUTPUT/scores"
"$TRAIN_PYTHON" scripts/research/mouth_expert/score.py
"$TRAIN_PYTHON" scripts/research/mouth_expert/summarize.py
"$TRAIN_PYTHON" scripts/research/mouth_expert/reference_check.py
```

Visually inspect a non-scored setup frame for the prescribed ROI before freeze;
never tune crops after seeing expert ranks. `render.html` contains only fixed
James views, no idle/head/blink motion. Source/weights/input hashes are frozen
before scoring; frozen JSON writes reject changed contents. Frame PNGs are
lossless and all comparisons retain exactly the same audio. `score.py` records
all raw AV loss components and SyncNet distances, with per-case features and
implementation controls. Its zero-input synthetic smoke test is not the James
compatibility result. Summaries apply the predeclared six-of-eight/per-engine
rules, never a best-per-corruption time shift.

The independent confirmation below was designed AFTER observing SyncNet's
secondary pass, and frozen before scoring a different corpus. It must not be
represented as part of the initial primary AVSR hypothesis. It uses the same
fixed−1-frame offset; repeat only under that explicit post-screen protocol.

```sh
"$TRAIN_PYTHON" scripts/research/mouth_expert/confirm_prepare.py
A2F_EXPERT_OUTPUT="$PWD/.local/mouth-expert-v1/confirmation" \
  node scripts/research/mouth_expert/render.mjs
"$TRAIN_PYTHON" scripts/research/mouth_expert/confirm_score.py
"$TRAIN_PYTHON" scripts/research/mouth_expert/audit.py
"$TRAIN_PYTHON" scripts/research/mouth_expert/plot.py
```

`audit.py` verifies source/audio/track/rig/checkpoint hashes, coefficient bounds,
geometry isolation, PNG dimensions, text separation and affected SyncNet windows.
Confirm the actual hosted blendshape download hash matches the frozen local
asset (the app uses its CDN URL) and retain `asset-audit.json`. The official
real-video example should recover offset3/confidence about10/min-distance5.35.
Do not confuse the minimum of per-frame distances with the minimum mean distance
across offsets; `reference_check.py` computes the latter.

Scores/figures are not training gradients or held-out v4 quality. These experts
remain offline and need not be shipped to the browser. No CDN publication or
Vercel actions are part of this diagnostic.

# Training-only SyncNet auxiliary-loss pilot

[Protocol](../../../docs/plans/compact-syncnet-loss.md).
Current v4 remains unchanged. New models, audio, dependencies and arrays stay in
ignored `.local/compact-syncnet-v1`. This single-seed pilot adds no browser
network, phoneme input, state, lookahead or expert download.

## Prerequisites

Use the immutable primary/replay feature caches, teacher rig assets and model
artifacts described in the data-scale and loss-balance reproduction guides.
Symlink those artifact directories into this dedicated worktree's `.local`;
never share writable `node_modules`. Also link `.local/mouth-expert-v1` from
the frozen mouth-expert compatibility experiment. Its official SyncNet source,
checkpoint, Python speech-features dependency and compatibility sets are required.
`common.py` names artifact directories and pins v4's hash.

Set `A2F_RESEARCH_WORK` and `A2F_TEACHER_ROOT` as in preceding experiments.
`EVAL_PYTHON` needs MLX0.31.2 / NumPy2.5.4 / ONNX1.23.2 / ORT1.31.1.
`TRAIN_PYTHON` needs PyTorch2.4.1 / NumPy2.2.6 / OpenCV4.12 / SciPy1.16.3.
`A2F_TORCH_SITE_PACKAGES` can point to the latter Python3.12 site-packages so
MLX training imports Torch for CPU scoring/export. Never use MPS for the expert.
Install app dependencies with Node22+ and `npm ci`.

Set `PLAYWRIGHT_MODULE` to Playwright1.65.0's `index.mjs` and
`WEBKIT_EXECUTABLE` to its installed WebKit2388 `pw_run.sh`. The actual Three
renderer is used, avoiding the rejected offline differentiable approximation.
No PyTorch3D install is needed for the final training method.

## Prepare and validate before student updates

Start `npm run dev -- --mode hosted` on its next free port and set
`DIAGNOSTIC_URL` to that actual URL. Launch the local renderer RPC in a separate
terminal; keep it running for calibration and the experimental arm:

```sh
export SYNCNET_RENDERER_ENDPOINT="$PWD/.local/compact-syncnet-v1/renderer-endpoint.json"
node scripts/research/syncnet_loss/browser_renderer.mjs
```

The RPC binds127.0.0.1 on a free port, recorded in that endpoint file. It renders
fixed head/gaze/blink only for the expert. The demo itself keeps animating idle.
No speech or model data leaves this computer.

```sh
"$TRAIN_PYTHON" scripts/research/syncnet_loss/cache.py
"$EVAL_PYTHON" scripts/research/syncnet_loss/browser_preflight.py
"$EVAL_PYTHON" scripts/research/syncnet_loss/calibrate.py
"$EVAL_PYTHON" scripts/research/syncnet_loss/prepare_confirmation.py
"$EVAL_PYTHON" scripts/research/syncnet_loss/freeze_training.py
```

The actual-render preflight checks pixel parity with the earlier compatibility
renders and numerical-gradient descent on eight first-batch training examples.
Freeze fails if this preflight or the backbone cotangent bridge fails. The
previous actual-render expert confirmation must also pass. Test examples do not
enter scale calibration. The final method does not require the rejected smooth
renderer to pass; its closure check failed and it is not used for training.

Audio embeddings cover28,845 primary windows. Mapping preserves exact60fps sample
positions and the previously fixed−40ms scorer alignment. The expert sees five
25fps mouth frames. Sixteen Rademacher central probes at epsilon0.003 estimate a
gradient on actual rendered coefficients. This stochastic finite-scale estimator
is not an exact analytic derivative. It is propagated through clipping and60→25
interpolation, then into the unchanged MLX backbone by the chain rule.

## Train, score and compare

```sh
"$EVAL_PYTHON" scripts/research/syncnet_loss/train.py --arm control
"$EVAL_PYTHON" scripts/research/syncnet_loss/train.py --arm syncnet
"$EVAL_PYTHON" scripts/research/syncnet_loss/run_confirmation.py
"$EVAL_PYTHON" scripts/research/syncnet_loss/audit.py
"$EVAL_PYTHON" scripts/research/syncnet_loss/build_review.py --freeze-only
"$EVAL_PYTHON" scripts/research/syncnet_loss/evaluate_experiment.py
"$EVAL_PYTHON" scripts/research/syncnet_loss/build_parity.py
"$EVAL_PYTHON" scripts/research/syncnet_loss/check_exports.py
"$TRAIN_PYTHON" scripts/research/syncnet_loss/build_review.py
```

Fresh synthesis/labeling may run independently before training finishes. Both
checkpoint selections must lock before its first score. Source hashes, sampler
hashes, optimizer/best/RNG checkpoints and audio/label provenance are preserved.
Auxiliary histories report actual expert loss separately from the cotangent
linearization used only to transfer gradients. Probe seeds37000+update and fixed
calibration seeds are independent of the main sampler. Resume recreates immutable
teacher-score caches without changing sampler state.

Stop the training renderer RPC and finish all training/labeling before browser
benchmarks. Follow the loss-balance browser guide, changing script paths to
`syncnet_loss`, output paths to `compact-syncnet-v1`, and model name `balanced`
to `syncnet`. Legacy environment names `A2F_LOSS_OUTPUT` and `LOSS_REVIEW_URL`
are retained by these measurement tools. All three models use the unchanged
production `/src/compact-webgpu.js`. Run three parity trials, three isolated
WebKit memory trials and session startup checks without other GPU jobs running.
Use the actual next-free server port and absolute Playwright/WebKit paths.

With those three browser environment variables set, the complete trial wrapper is:

```sh
mkdir -p .local/compact-syncnet-v1/memory
clang -O2 scripts/research/compact_ablation/sample.c -o .local/compact-syncnet-v1/memory/sample
node scripts/research/syncnet_loss/run_browser_checks.mjs
"$EVAL_PYTHON" scripts/research/syncnet_loss/analyze.py
```

It saves three independent parity/memory trials, checks byte-identical resets,
measures session startup and verifies all sixteen synchronized review cases.

Memory sampling sums WebContent+GPU physical footprint every20ms with the real
James rig and continued idle animation. It uses accelerated prerecorded STA and
excludes TTS, initial rig loading, network and browser-parent processes. It does
not establish full-pipeline memory on a physical iPhone. Run `analyze.py` only
after collecting its expected three measurement trials. No CDN upload or Vercel
deployment is part of this experiment.

## Rejected renderer setup

`render.py`, `preflight.py`, `confirm_renderer.py`, `auxiliary.py`, neutral capture
helpers and ignored renderer-trial folders preserve the earlier unsuccessful
analytic-gradient approach. The final smooth renderer's derivatives pass, but
only4/8 missing-closure comparisons do. It must not replace the actual renderer
in training based solely on differentiability or speed.

To repeat that rejected setup only: CPU PyTorch3D0.7.9, official commit
`33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`, was built with Apple clang into
`OUT/vendor-deps` using pip's `--no-build-isolation --no-deps --target` options.
Local dependencies: iopath0.1.10, fvcore0.1.5.post20221221, yacs0.1.8,
portalocker3.2.0, termcolor3.1.0. Capture the fixed neutral photo with
`capture_neutral.mjs`, then `prepare_neutral.py`. Geometry/depth use float64;
training images would have been float32, with float64 finite-difference checks.
A ragged pixel reduction improved a five-frame gradient calculation from3.2s
to0.5s without changing its score, but did not resolve closure compatibility.

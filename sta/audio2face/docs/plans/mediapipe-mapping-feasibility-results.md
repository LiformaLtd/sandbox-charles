# ARKit to MediaPipe mapping feasibility results

Date: 2026-11-07. Status: **mapping not verified; do not change the v1 player path**.

The standalone probe used actual avatar viseme meshes, actual output from the pinned compact network, and the matching source rig. The first residual-linear mapping does not pass its numerical quality screen on any of the 11 tested fits. This rejects the current calibration recipe as ready for integration; it does not establish that a robust mapping is impossible.

## What was tested

- Eleven independently loaded fits: eight contrasting avatars plus the current and two historical fits of avatar `05a87620`. The set deliberately includes a naturally parted neutral mouth, stylized human faces, nonhuman faces, and an anime face. These are a stress set, not a random estimate of all avatar quality. Historical fits are used only as fit-variation tests, not asserted to be current published avatars.
- Seven calibration roles per fit: PP, aa, E, ih, oh, ou and FF. Cohort landmarks were used where present, with stable upper-face alignment into that fit's own reference. Original source-image landmarks remain separate.
- Source examples selected from 48 recorded Trevor/Pippa clips. Each example is an actual predicted vector, not an averaged or guessed ARKit vector. Selection is currently automatic by phone role plus measured source geometry; these assignments have not passed independent source-pose review.
- Forty-eight M3/F3 clips held out of mapping-example selection: 11,330 frames at 60 fps, about 188.8 seconds. Held out here refers to mapping calibration, not an assertion about all original neural-network training.
- Source reference coefficients measured from silent PCM. Its mapping to each target reference is exact. Only jaw, mouth and cheek-puff controls can affect destination speech geometry; eye and brow controls are excluded.
- Pinned ONNX SHA-256 `b4a275acf78b982fb6ec6fc55333c5cf25613e72b6d9b3d80e95d6e74ac6261d`. A fresh full-clip replay matched cached coefficients exactly (maximum absolute difference 0.0). The sandbox's newer active export was not silently substituted.

## Measurements

All landmark errors and correction magnitudes below are fractions of the target fit's neutral mouth width, expressed as percentages. The primary regularization parameter was fixed at 0.03. The 0.0001 arm is an explicitly exploratory sensitivity check, not a revised acceptance threshold.

| Fit | Primary calibration RMS max | Primary held-out viseme RMS max | Low-regularization calibration RMS max | Low-regularization held-out viseme RMS max | Primary maximum contact correction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01d55f04 | 11.6% | 22.8% | 0.18% | 29.7% | 7.2% |
| dd80126a | 13.6% | 26.7% | 0.23% | 38.3% | 9.1% |
| 0c72c586 | 9.7% | 19.0% | 0.24% | 21.1% | 6.5% |
| 3bbd86ff | 7.7% | 13.5% | 0.26% | 17.0% | 6.7% |
| 394d2cb9 | 10.8% | 21.3% | 0.26% | 22.3% | 13.2% |
| c1bb51a7 | 32.2% | 63.3% | 0.36% | 64.5% | 15.0% |
| d3194283 | 12.7% | 29.4% | 0.35% | 41.6% | 15.0% |
| 64cf2cc1 | 8.9% | 18.2% | 0.29% | 21.2% | 8.7% |
| 05a87620 | 6.8% | 13.4% | 0.15% | 13.8% | 4.3% |
| 05a87620_old | 7.2% | 14.2% | 0.15% | 14.6% | 4.3% |
| 05a87620_old2 | 6.9% | 13.7% | 0.15% | 14.0% | 4.3% |

The fixed limits are 3% calibration RMS, 6% held-out viseme RMS and 5% maximum contact correction. No fit passes the complete screen. The source example matrix has rank 7; it cannot identify 52 independent expression controls. Unmeasured directions retain the source deformation prior.

![Calibration and held-out errors](../../.local/mediapipe-mapping-pilot-v3/mapping-errors.png)

Reducing regularization reproduces the selected calibration poses within 0.4% of mouth width, but held-out viseme errors remain large. Fitting the known poses therefore does not establish generalization. The selected aa and E source poses also have almost the same normalized aperture, while many target meshes express very different aa and E openings. This is direct evidence that a viseme label and one representative speech frame are insufficient to define a dependable coefficient-to-target contract.

## Lip contact and asset quality

The candidate contact constraint brings 99.7–100% of eligible active source-seal frames within the target gap limit. However, it can require 7–15% of a mouth width of movement on several fits, exceeding the predeclared 5% budget. Remaining penetration occurs on some other frames. Forcing lips to meet does not make the rest of the deformation correct, and is not accepted as evidence that the mapping works.

Contact eligibility excludes coefficient vectors within 0.15 of the source reference, measured over owned channels. Those reference-like frames have a separate rest-preservation requirement: a naturally parted target reference must not be forced shut. The held-out tracks contain 1,263 eligible active seal frames and 1,333 reference-like source-seal frames. This is a geometric consistency check against the source rig, not independent lip-sync ground truth.

The existing PP meshes for `dd80126a`, `394d2cb9`, `c1bb51a7`, `d3194283`, `64cf2cc1` exceed the chosen inner-lip gap or penetration tolerance. Some misses are marginal; the larger gaps need inspection. A PP label alone is not accepted as a closed-contact target. No original mesh was repaired or overwritten.

## Visual comparisons

Green contours show each actual stored viseme mesh; red contours show the primary mapping prediction at its assigned source coefficients. The background is the existing source viseme image. Cohort predictions are transformed back using stable face anchors before overlay. These are geometry comparisons, not altered textures or a new player renderer.

![Actual fitted visemes versus mapped contours](../../.local/mediapipe-mapping-pilot-v3/viseme-fit-comparison.png)

The source library poses below are rendered directly from the matching source rig. They remain candidate assignments requiring review, especially coarticulated vowels, rounded controls and the labiodental role.

![Observed source calibration poses](../../.local/mediapipe-mapping-pilot-v3/source-calibration-poses.png)

## Cost and limits

The residual solve and ownership mask alone took 0.241–0.603 ms per face in this Python/NumPy run on the current Mac. This excludes file reads, JSON decoding, alignment, source registration, model startup and per-frame constraints, and is not a browser or mobile startup benchmark. It supports treating the small solve as a modest cost, but quality remains unresolved.

The source registration uses a synthetic front render and fixed barycentric embedding, with nine explicitly paired source lip locations. Other landmark registration is approximate and needs independent review. Only image-plane fitting has been measured. Full mesh triangle safety, depth, smoothing, texture warping and the production blink/oral compositor have not passed a gate in this probe. No runtime integration is authorized by these results.

## Recommended next experiment

1. Review and improve source pose correspondence before any player refactor. Replace one automatically selected frame per phoneme role with a small geometry-conditioned library spanning independent aperture, spread, rounding, lip roll/press and jaw-with-lip-closure combinations. Retain actual model sequences as the final distribution check. Calibrate shared coefficient semantics on the source rig rather than choosing more regularization to hide poor examples.
2. Validate the canonical source mouth registration and its local frames with controlled coefficient sweeps. Separate uncertainty in transferred anatomy from uncertainty in coefficient assignment.
3. Select reliable actual contact/open/spread/rounding targets for each fit. Inspect the PP outliers and use another verified same-face closed pose where available; do not edit avatar masters during this experiment.
4. Compare a geometry-matched residual fit with a bounded local nonlinear residual/interpolation model if the linear fit still fails. Freeze the candidate and thresholds before evaluating independent clips/faces. Held-out target meshes must not be used to select their own test coefficient inputs.
5. Keep the player geometry refactor and ONNX integration blocked until a candidate passes reconstruction, interpolation, contact, correction-size and geometry-safety checks. Later, build the same verified mapping in the browser initially, then move it to avatar publishing as already agreed.

## Reproduction and preservation

Research branch: `research/mediapipe-mapping-feasibility` in the dedicated sandbox worktree. Scripts and run instructions are included in `mapping-feasibility-probe.zip`. Raw results, the fixed protocol, source pose definitions, input hashes and sensitivity results are included separately. The earlier exploratory v1/v2 runs exposed ownership/display and rest-eligibility issues; the report uses the corrected, explicitly versioned v3 protocol. Numerical quality limits were not relaxed.

Five invariant tests cover residual reconstruction, preservation of nonzero source reference and unobserved channels, contact direction/input preservation, coordinate round trips/stable alignment, and exact short/tail audio-window coverage. Cached ONNX predictions were checked against an exact replay. The original avatar meshes and pinned model checksums are rechecked after the run.

No player, TTS service, CDN publisher, source avatar mesh or deployment was changed. The primary player checkout already had unrelated location UI edits when inspected; this work does not modify or revert them.

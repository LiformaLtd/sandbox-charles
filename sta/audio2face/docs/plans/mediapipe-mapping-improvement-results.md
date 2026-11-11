# Geometry matched mapping and noisy mesh calibration results

Date: 2026-11-07. **Research improved, but the mapping is not cleared for v1 integration. The player and original avatar assets remain unchanged.**

## Goal and quality standard

The objective is coherent speech motion on each face, not exact reproduction of every old mesh. Existing viseme fits are noisy observations. Clear fitting defects can be corrected when the correction is supported by anatomy, neighboring landmarks or consistent poses. Face proportions, asymmetry and naturally parted rest must remain recognizable. A visually plausible but wholly incorrect fit cannot be identified reliably from geometry alone; it needs image evidence or a trusted independent observation.

Reconstruction error against raw meshes is therefore reported separately from error against anatomically constrained calibration targets. Matching an erroneous raw PP gap is not treated as success. Conversely, neither a small mean error nor a forced lip seal is enough to approve a mapping.

## Work completed

- Replaced phoneme-role representative source frames with bounded source-rig coefficient fits matched to actual mouth geometry: detailed lip displacement, aperture profile, spread and upper/lower lip motion. Source assignments use four development faces only. Eye and brow controls remain excluded from speech geometry.
- Tested the same original mesh fits, then expanded the stress set to **19 fits**. Up to 22 existing viseme/expression examples are used per fit; absent auxiliary examples are recorded, not silently substituted. Three versions of one avatar are included.
- Corrected an error in the probe's source correspondence. The nine verified source lip-pair samples cover a central portion of the mouth; assigning them directly to nine MediaPipe indices compressed the inner contour. The new registration resamples the source lip curves at anatomical locations across the full mouth. The source model itself was not changed.
- Added robust upper-face alignment, isolated contour anomaly handling, bounded closed-pose corrections and point-wise robust fitting. Original photographs, UV/source landmarks and asset files are never overwritten. Statistical contour repairs remain proposals requiring visual review, especially on stylized faces.
- Tested injected errors using in-memory copies of actual fits: a lip landmark displaced by 25% of mouth width, an artificial 8% closed-mouth gap, 1% Gaussian lip noise and one damaged alignment anchor.
- Tested a local nonlinear residual with a fit-specific anatomical prior. Its parameters were selected using only the four development faces before the remaining evaluation fits were processed. It did not solve the withheld-pose problem and is not accepted as a production replacement.

All speech checks reuse the same 48 M3/F3 clips (11,330 frames) held out of mapping-example selection and the originally pinned compact ONNX export. Source seal labels still come from the original nine verified source rig pairs, so correcting registration does not change that label definition. The test sets have been reused across exploratory iterations; these are engineering comparisons, not a blinded final quality trial.

The robust residual solve took 7.6–19.9 ms per face on this Mac, compared with less than 1 ms for the original simple linear solve. This excludes asset loading, model startup and browser/mobile overhead. Robust calibration adds work, but it is still a once-per-fit operation; these figures are not a browser latency promise.

## Geometry matching improvement

In a controlled assignment-only comparison using the same seven core poses, solver and previous registration, the seven common evaluation fits outside the development set showed:

- Mean worst calibration error: 12.22% → 2.55% of neutral mouth width, a 79% reduction.
- Mean worst withheld-viseme error: 24.77% → 16.00%, a 35% reduction. This remains above the 6% screen.

This isolates the benefit of improved source assignment. It is not the final quality score of the later robust pipeline, which changes both correspondence and observation treatment.

![Controlled geometry matching comparison](../../.local/mediapipe-mapping-improvement-report/geometry-matching-comparison.png)

## Source correspondence correction

On the same held-out coefficients, the previous source registration inverted the projected inner/outer lip thickness ordering on 43.2% of source frames. The corrected registration reduced this diagnostic to 0.0%. This fixes an error in the research mapping rather than blaming it on an avatar or the audio model. Full 3D correspondence outside the mouth still needs review.

![Previous and corrected source lip correspondence](../../.local/mediapipe-mapping-improvement-report/source-registration-comparison.png)

## Correcting existing closed-mouth fitting defects

The calibration now estimates a closed contact shape instead of requiring an imperfect PP mesh to be an exact target. It preserves the fitted lip midline and moves the upper/lower boundaries together with their outer rims, retaining thickness rather than squeezing lips into a line. Repairs are bounded and recorded separately from runtime deformation. Parted neutral mouths are not forced closed.

For the largest observed PP gap, the measured aperture was reduced from approximately 6.2% to 0.3% of mouth width with about 3% movement per lip. This is a concrete geometric repair in the calibration target, **not yet proof of better complete textured speech animation**.

Green shows original fitted contours; red shows the proposed closed-pose calibration repair. The photographs and textures are unchanged and are not warped in this comparison.

![Examples of closed-mouth calibration repairs](../../.local/mediapipe-mapping-improvement-report/closed-mouth-repairs.png)

| Actual fit | Raw PP maximum gap | Repaired calibration gap | Worst withheld-pose RMS in robust linear probe |
| --- | ---: | ---: | ---: |
| 01d55f04 | 0.22% | 0.30% | 10.9% |
| dd80126a | 1.02% | 0.30% | 7.9% |
| 0c72c586 | 0.80% | 0.30% | 5.5% |
| 3bbd86ff | 0.31% | 0.30% | 7.9% |
| 394d2cb9 | 1.01% | 0.30% | 13.8% |
| c1bb51a7 | 2.12% | 0.30% | 52.1% |
| d3194283 | 4.13% | 0.30% | 31.9% |
| 64cf2cc1 | 6.19% | 0.30% | 10.5% |
| 05a87620 | 0.20% | 0.30% | 7.9% |
| 05a87620_old | 0.20% | 0.30% | 15.5% |
| 05a87620_old2 | 0.20% | 0.30% | 9.1% |
| 5a02aa84 | 0.60% | 0.30% | 6.2% |
| c4e71fe5 | 0.47% | 0.30% | 19.3% |
| e76f3815 | 0.43% | 0.30% | 9.3% |
| ff5687da | 0.60% | 0.30% | 12.8% |
| 201c8934 | 1.05% | 0.30% | 8.2% |
| 679803f3 | 0.81% | 0.30% | 7.0% |
| 8f228898 | 0.80% | 0.30% | 9.4% |
| dac8814d | 0.75% | 0.30% | 14.1% |

The last column is an engineering diagnostic against constrained observations, not a declaration that every withheld old mesh is ground truth. Remaining failures also include motion-induced thickness loss and large localized responses, so raw reconstruction scores are not the sole reason integration remains blocked.

## Robustness measurements

The median influence of an injected isolated lip-vertex error fell by approximately 94%. Added false closed-mouth gaps were largely removed by the contact constraint. Damaging one alignment anchor no longer moves the entire mouth substantially. All 19 fits stayed within the 3% average lip-RMS stability screen for the chosen corruption cases.

However, average stability hides localized failures: the worst individual point still changed by about 18% of mouth width in a stress case. Gaussian noise was not improved on average. The robust calibration therefore **has not passed a complete robustness gate**. Strong smoothing or rejecting a distinctive whole pose would not be an acceptable way to hide this failure.

The image-plane triangle-orientation check is deliberately conservative. Some original valid-looking fits already reverse tiny projected triangles relative to neutral, so that count alone is not proof of a rendering defect. It is retained alongside the source-asset baseline and lip-thickness results for review. The actual production depth/oral/blink compositor has not been modified or used to claim a pass.

## Nonlinear trial and remaining work

The local Gaussian residual fits calibration examples almost exactly and returns toward an anatomical prior outside the calibrated region. It still misses the 6% withheld-pose screen across the 19-fit set. More fitting flexibility alone is therefore insufficient.

The next iteration should establish image-grounded confidence for questionable observations and validate cross-pose source/target correspondence, especially rounded vowels and stylized mouths. Keep bounded contact and thickness constraints, add pointwise stability limits, and verify temporal continuity. Use reviewed motion/appearance comparisons as well as geometry diagnostics. A mapper should improve a clear defect, retain genuine facial character and flag unsupported ambiguity; it should not blindly copy bad fits or silently invent a different face.

No player refactor, selector wiring, model deployment or default switch should proceed from these results. Browser calibration remains the initial implementation location after verification; publish-time mapping remains the agreed later architecture.

## Reproduction and preservation

The research branch is `research/mediapipe-mapping-feasibility` in its dedicated sandbox worktree. The archive contains scripts, pinned requirements, protocols, results and run instructions. Numerical limits were not relaxed to obtain a pass. Sixteen invariant tests pass; cached coefficients retain the original exact replay check. Original mesh and model hashes and the complete Player Git diff are checked before delivery.

The primary Player checkout contained unrelated location UI changes at the start of this work. They are preserved. No v1 player code, original avatar source mesh, TTS endpoint, CDN publisher or deployment was changed.

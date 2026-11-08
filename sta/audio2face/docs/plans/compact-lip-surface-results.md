# Dense lip-surface supervision: results

**Dense surface supervision gave a small extra improvement, with a closure
trade-off.** Relative to matched ordinary fine-tuning, original-level 3D lip
surface RMS improved **1.23%**, and nine-gap RMS improved **0.41%**. Both engines
improved on both metrics. Three well-timed closures were lost versus control.
The fixed pilot missed its ≥3% surface-gain screen and its closure-count guard.
Keep published **v4 finetune-r4**; this model was not uploaded or deployed.

Read the [frozen plan](compact-lip-surface.md) and
[reproduction instructions](../../scripts/research/lip_surface/README.md).
Core protocol/code commit `58954f1` preceded optimization and fresh scoring.
This is agreement with full Audio2Face, not filmed-human ground truth or a
blinded perceptual quality result.

## Dense target, same browser model

A fixed spatial mouth/perioral region covers **2,267 vertices / 6,801 coordinates**,
including all 18 original gap landmarks, lateral corners, inner/outer mouth skin
and nearby skin. Bounds on neutral teacher skin are abs(x)<3, 166.1<y<169, z>5 cm,
matching the earlier fitter's mouth bounds. It is not a manual vermilion mask.
Only 1,077 selected vertices lie in the older frontal fit mask; the new direct
surface term supervises the entire selected region. The mask plot is retained
as `.local/compact-surface-v1/surface-region.png`.

Rig prediction is `R+wB`, teacher is `S+D`; target relative to rig neutral is
`t=D+(S-R)`. The neutral offset and matching vertex/channel coordinate frames
were checked against the established gap rig. The extra target uses raw teacher
geometry, not only fitted coefficients or nine vertical gaps.

Per axis, cache `q=tBᵀ/V` and `s=mean(t²)` with fixed `G=BBᵀ/V`. Then
`wGw−2wq+s` exactly equals dense per-axis MSE. Summing axes gives mean squared
3D vertex distance. Keeping the constant term preserves the residual outside
the rig span. This is exact sufficient-statistic compression, with float32
roundoff, not PCA or dropping surface detail. No new neural parameters are added.

Each target frame needs 120 float32 values instead of 6,801 coordinates. The
35,262 cached training/development windows occupy about 508 MB in this form,
versus about 28.8 GB for corresponding float32 dense-coordinate maps. These are
offline maps; none is exported to the browser. Absolute head-height coordinates
are not quantized into float16.

An offline backend copy projects the teacher output before host transfer while
preserving the original graph, recurrent state, priming windows, identity, seed
and crop. Against 3,997,395 untouched-teacher values, projected statistics differed
by at most 1.55e-6 and the nine gaps matched exactly. Dense/quadratic real-frame
loss and gradient max deltas were 2.03e-6; CPU/MLX new-loss delta 7.45e-9 and
backbone gradient relative L2 difference 3.34e-7. Three tests verify dense-error
identity, neutral-coordinate cancellation and complete original-landmark coverage.

## Data and matched optimization

No new training speech was added. Reuse the audited expanded primary corpus:
3,328 original recordings / 3,840 gain-conditioned examples plus unchanged 549
clean replay clips. Regenerate surface statistics for those 3,840 primary cases
and 864 common-development cases. Existing animation labels and mel maps stay
unchanged. All 4,704 regenerated gap tracks matched their existing references
exactly; waveform hashes, timestamps and recipe/source hashes were checked.

Fresh confirmation has 24 entirely new disjoint texts: 12 authored closure/vowel/
rounding examples and 12 unused CMU ARCTIC sentences. Four previously studied
training-held-out presets produce 96 new recordings / 384 independently labelled
original/half/headroom-boost/quarter cases. Prior inventory excludes 2,653 retained
texts, including the targeted-speech experiment. These are new texts/recordings,
not new voices. Fresh surface tracks also match all pinned gaps exactly. Source,
license, synthesis, aligner, teacher and fitter provenance are retained. No
original human speech is reused.

Initialize both arms from released v4 SHA
`ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`.
Both keep 331,047 parameters, normalization, ordinary losses, primary/replay
batches, seed 37, 3,720 AdamW/cosine updates and identical sampler hashes. Control
surface weight is 0. Surface supervision applies to 48 primary windows per update;
16 clean replay windows retain ordinary animation losses only.

One frozen training minibatch sets added gradient norm to 25% of main-objective
norm. Main norm was 1.02599, raw surface norm 0.39472, giving coefficient
**0.64982123** on mean summed-axis MSE in cm², ramped over 372 steps. Gradient
cosine with the ordinary objective was 0.4493. No coefficient sweep or test tuning
was performed. The initial batch surface MSE was 0.04698 cm²; this is a training
diagnostic, not held-out accuracy.

Both arms use the SAME NEW joint development selector: gap RMS + 0.5×dense 3D
surface RMS + 0.25×(1−contact F1) on four common-dev voices; the Easton guard
contributes its gap/contact score as fifth voice, with no surface target there.
RMS terms use centimetres inside selection. This differs from earlier gap-only
selection but is shared by the matched arms. Initial development results matched
exactly; both selected step**2976**. Initial checkpoint was eligible. Selected
exports, source/map/teacher locks and visual cases were fixed before fresh scores.

## Fresh quality

| Model | Dense 3D lip RMS mm ↓ | Nine-gap RMS mm ↓ | Contact F1 ↑ | Opening RMS mm ↓ | Well-timed /m b p/ ↑ |
|---|---:|---:|---:|---:|---:|
| Released v4 finetune-r4 | 2.89763 | 2.46046 | 0.79491 | 2.77468 | 162 |
| Matched fine-tuning control | 2.84445 | 2.41529 | 0.79983 | 2.71075 | 164 |
| Dense surface supervision | 2.80955 | 2.40532 | 0.79948 | 2.69420 | 161 |

All rows use the same fresh cohort and macro-voice aggregation. Dense3D surface
RMS is a different measurement from lip-gap RMS; do not compare 2.80955 with
an older gap score as though that were an accuracy improvement. Current v4's
previously reported scores on other cohorts remain valid for those cohorts.

Surface supervision improves full3D RMS 3.04% versus released v4, but only 1.23%
versus control. Shared extra fine-tuning accounts for part of the gain. Gap RMS
improves 0.41% versus control. Engine-specific control-relative surface gains:
Phonon 1.15%, Supertonic 1.30%; gap gains 0.285% and 0.534% respectively.

| Surface component RMS mm | Released v4 | Control | Surface |
|---|---:|---:|---:|
| Lateral X | 0.81893 | 0.81406 | 0.80661 |
| Vertical Y | 1.54931 | 1.52645 | 1.49929 |
| Depth Z | 2.30726 | 2.25749 | 2.23437 |

These are pointwise displacement components, not scalar width/protrusion
measurements. Depth contributes the largest measured component error. The
existing teacher-fitted rig reference has **0.85555 mm** dense 3D RMS on original
speech. That is a retargeting-residual reference, not a strictly optimal lower
bound. Student prediction remains the larger measured error stage here; this
does not prove the reason or establish that all remaining error is learnable.

| Gain | Dense3D RMS: v4 / control / surface (mm) | Gap RMS: v4 / control / surface (mm) |
|---|---|---|
| Original | 2.89763 /2.84445 /2.80955 | 2.46046 /2.41529 /2.40532 |
| Half | 3.07767 /3.01008 /2.98498 | 2.60762 /2.56586 /2.55567 |
| Headroom boost | 2.87892 /2.83849 /2.79953 | 2.48797 /2.45047 /2.44176 |
| Quarter | 3.46096 /3.40812 /3.39333 | 2.94258 /2.90819 /2.89526 |

Well-timed /m/, /b/, /p/ counts are 45/63/54 for release, 43/67/54 for control and
43/65/53 for surface. The total falls 164→161 versus control (release 162). Frame
contact F1 declines slightly versus control, within the predeclared allowance.
Existing anticipatory event association and one-to-one matching remain unchanged.
Full per-phone onset/release, ambiguity/missed-event, engine/voice/category and
velocity metrics are retained; conditional timing must be read with event counts.

Historical gap RMS/F1: release 2.54635/.77603; control 2.49198/.78297;
surface 2.48465/.78071. Inspected feature-cohort gap RMS/F1: release 2.42607/.80250;
control 2.38722/.80882; surface 2.37885/.80920. These are regression checks,
not fresh confirmation; dense historical surface was not regenerated.

The primary ≥3% extra surface gain and total-closure guard fail. All other
predeclared guards pass. Preserve `promising=false`. This fixed coefficient,
region and seed show a modest useful direction, not a large quality improvement
or proof that every surface-loss weighting would behave similarly. No tuning
was performed against this inspected confirmation set.

## Browser speed and memory

All three exports retain **2,251,827 bytes / 331,047 parameters**, identical ONNX
operator topology, **25 buffers / 1,492,236 explicit model GPU bytes** and two
reusable compute pipelines. Geometry/statistics/Gram loss tensors are absent
from the browser model; the PCM input and 52 output channels are unchanged.

Twelve complete native-ORT/WebGPU fixtures × three models × three fresh runs,
including first-clip repeats, give 117 comparisons. Maximum coefficient delta
was 9.54e-7, with no nonfinite/unexpected zeroed frames. Repeated first-clip GPU
output is byte-identical for every model. Sixteen synchronized James cases
passed audio and exact seeks; the four-lane review includes both gap and dense
surface-error plots. Functional review and representative screenshots are not
a blinded perceptual study.

| Model | Warm packed-block median ms | Packed medians range ms | Sampled WebContent+GPU peak MiB |
|---|---:|---:|---:|
| Released v4 | 1.04 | 0.98–1.28 | 532.4 |
| Control | 1.13 | 0.98–1.46 | 525.3 |
| Surface | 1.02 | 0.96–1.06 | 525.7 |

Timing is the median of six packed-track medians across three settled desktop
WebKit runs, after rig readiness and five warm-up calls. Full blocks emit 500 ms
of animation audio. These are desktop M5 Max STA measurements, not TTS or iPhone
latency. Overlapping timing variation does not establish a speed change for this
unchanged graph.

Physical footprints sum new desktop WebKit WebContent+GPU processes concurrently,
sampled every 20 ms during short/long/repeated prerecorded STA alongside the normal
James idle animation. **TTS is excluded.** Automation uses an iPhone 13 viewport/UA on desktop WebKit.
Single bounded peaks and process/
driver variation do not establish a memory saving. Browser/network processes
are excluded and brief spikes may be missed. No full-pipeline or physical-iPhone
under 2 GB guarantee is made.

## Artifacts and decision

Ignored `.local/compact-surface-v1` retains mask/rig/source provenance, exact
teacher-projection checks, all 5,088 case-reference checks, aligned maps, calibration,
source/training/selection locks, model/optimizer/RNG checkpoints, full metrics,
fitted references, browser parity/timing/process traces and `review.html`.
The local review used http://127.0.0.1:5184/.local/compact-surface-v1/review.html;
use Vite's printed next free port after restart.

Training-protocol SHA: `85219f3bb2a70a75d81ed4a4ff5e7ab9b13a5da32b485527efd1a7a1e9a8434a`.
Control: `63ddc0c64851edc445eba1364e8b160a8a680c4d3b66638fc57a17b782cfe867`.
Surface: `45ee59924eff2c14b0c6e40682afe29f7c6086f1ff5a99b41abbcbd1dff2d317`.

Keep deployed finetune-r4 and its WebGPU default. Any further weight, mask,
context or representation change requires a separate controlled experiment and
new confirmation; do not optimize this now-inspected test into a release win.

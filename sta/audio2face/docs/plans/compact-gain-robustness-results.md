# Teacher-labelled gain follow-up — results

Protocol: [compact-gain-robustness.md](compact-gain-robustness.md).
Reproduction: [research README](../../scripts/research/gain/README.md).

## Initial research decision

The first-seed gain arm did **not** pass the predeclared 5% improvement gate.
Its mean lip-gap RMS on half-level and boosted held-out speech improved **3.4%**
versus a matched original-only fine-tune. Extra-quiet speech improved more,
but ordinary-level quality changed only slightly. The initial recommendation
was to keep released v4 unchanged and not advance this arm to the planned
independent-seed/fresh-confirmation phase. The later user-requested publication
below changes the deployment decision, not these measurements.

## Controlled experiment

Both arms started from the same frozen v4 representative, seed 37, 24 epochs,
31 updates/epoch, existing losses and clean replay. The augmented arm changed
only the speech-level distribution. Both selected checkpoints against the same
gain-varied development cohort plus the clean Easton guard. Control selected
epoch 19; augmentation selected epoch 15. Model hashes and selections were
saved before held-out scoring. Each export is **2,251,827 bytes**, with
**331,047 learned parameters** and unchanged inference topology.

The 336 pilot recordings produced 1,056 original/transformed recordings:
original, 0.5× and headroom-capped boost for every recording, plus 0.25× only
for the 48 held-out recordings. Training had 256 original versus 768 augmented
recordings; both arms had the same 96 development recordings. Evaluation had
192 recordings, covering four conditions and four held-out voices. No new
distinct sentences were added. The existing fresh v4 confirmation cohort was
not used for selection or this experiment.

Boosts ranged from 1.142× to 2×; 129 of 336 were limited by headroom. None was
clipped. Maximum waveform peak was 0.980000019 (float32 rounding). All labels,
including originals, were regenerated under one labeler fingerprint. Original
weights and raw gaps matched previous labels exactly. Raw teacher gap changes
were 0.286 mm RMS for half-level, 0.246 mm for boost, and 0.469 mm for quarter
level; these are broader-corpus figures, not the earlier 16-clip diagnostic.

## Held-out quality

RMS below is the mean per-voice error against nine raw teacher lip gaps,
in millimetres. This is teacher agreement, not a human preference score.

| Audio level | Released v4 | Matched control | Gain arm | Gain improvement vs control |
| --- | ---: | ---: | ---: | ---: |
| Original | 2.7810 | 2.7448 | 2.7359 | 0.3% |
| Half | 3.1560 | 3.1367 | 3.0227 | 3.6% |
| Headroom-capped boost | 2.8851 | 2.8742 | 2.7831 | 3.2% |
| Quarter, absent from training/development | 3.8572 | 3.8714 | 3.5891 | 7.3% |

The primary two-gain reduction was 2.6% for Phonon and 4.3% for Supertonic.
Both engines improved, but the aggregate 3.4% failed the 5% threshold. Original
RMS was 1.6% better than released v4; much of that was also obtained by the
matched control's extra fine-tuning, so it cannot all be attributed to gain data.

Original-level contact F1 was 0.7525 for v4 and approximately 0.7607 for both
fine-tunes. Half-level F1 improved 0.7500 → 0.7574 versus control; boosted F1
0.7433 → 0.7624; quarter-level F1 0.7126 → 0.7296. The ordinary-level no-regression
checks passed. Opening-gap RMS on half-level speech improved 3.7444 → 3.5854 mm
and quarter-level 4.7241 → 4.3407 mm versus control, but boosted openings and
ordinary openings were slightly worse than control. Gains are not uniform.

For the two trained gain conditions together, well-timed closure hits versus
control were `/m/` 43 → 50 of 88 eligible teacher episodes; `/b/` 68 → 76 of
102; `/p/` 80 → 89 of 123. These use the existing unique association, corrected
100 ms lead audit and actual teacher onset/IoU criteria. Teacher contact can
change with gain, so episode denominators are recalculated, not copied.

Historical 48-clip Easton RMS was 2.583 mm for v4, 2.549 mm for control and
2.584 mm for augmentation. Gain-arm F1 was 0.7754 versus v4's 0.7720. This is
effectively a tie on RMS against v4, not a meaningful Easton quality improvement.

## Review and limitations

Twelve cases were chosen by voice/category/condition before held-out scoring.
The four-lane viewer synchronizes fitted teacher, v4, control and gain arm on
the same audio/frame clock. WebKit verification passed every case's audio and
exact snapshot seek. Selected rendered contact/opening snapshots were inspected;
they do not show a broad leap toward teacher detail. This is not a blinded study.
Open the generated `.local/compact-gain-v2/review.html` through the Vite server.

This is one seed and a previously inspected pilot voice/text cohort. A larger
confirmation claim is not justified. Because the arm missed its initial gate,
it was not promoted, given additional seed sweeps, or subjected to a new browser
speed/memory release benchmark. Equal graph size/work does not constitute a new
physical-iPhone memory measurement. The original study did not change demo
weights or idle animation; the subsequent publication is recorded below.

The next main quality experiment should broaden distinct lexical/acoustic
coverage and test independent representation capacity, rather than keep tuning
gain settings against these held-out recordings. Gain variation is useful
robustness data for a later recipe, but has not explained the larger fidelity gap.

Ignored artifact root: `.local/compact-gain-v2`. It contains protocol/source
locks, all audio/teacher labels, input audit, feature caches, complete training
checkpoints/logs, frozen selection, per-clip evaluation and visual-review records.

## User-requested v4 update

After reviewing these results, the user explicitly requested that this arm
replace the active v4 and be published. It is released as **v4 gain-r1**, still
shown as “Compact ARKit v4 (2.25 MB)” in the demo. The initial 5% study gate
remains recorded as failed; no additional-seed confirmation is implied.

Immutable model URL:
https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/arkit-student-v4-gain-r1/james-arkit-student-v4.onnx

SHA-256: `84da9bfe22d221075d85e8a52622fc7d31b18d12b04905e0902a7ec64eb5ab2f`.
The prefix also contains the upstream licenses, notices and checksum manifest.
The earlier `arkit-student-v4-candidate` prefix and its seed-73 weights are
preserved for rollback; the new revision never overwrites the old CDN objects.

Before publication, full WebGPU/native comparisons covered 279,708 coefficients
across two engines and original/half/boost/quarter levels, plus both full recorded
“packed” passages. Maximum absolute error was 7.75e-7 with no zeroed or non-finite
frames. The browser graph, WebGPU default, head motion and continuous idle
animation are unchanged. Physical iPhone review remains separate.

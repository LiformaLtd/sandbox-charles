# Current v4 memorisation diagnostic

Completed 8 November 2026. **The unchanged architecture can fit this tiny known
set close to its fitted reference.** These are training-set results, not improved
held-out quality, and no weights were published.

[Protocol](v4-memorisation.md) ·
[Reproduction](../../scripts/research/memorisation/README.md)

## Why repeat this?

An earlier four-clip data-scale diagnostic already succeeded: about 0.42 mm
raw-teacher error versus a 0.372 mm fitted reference. The recommendation to run a
memorisation test again should have accounted for that existing result. This
study rechecks current v4 finetune-r4 on a different, slightly broader tiny set;
it does not establish learnability for the first time.

Eight original-level training recordings were selected by salted transcript hash,
one per Phonon Alfie/Easton/Maeve/Zoey and Supertonic F1/F2/M1/M2, with distinct
texts and no error-based selection. The set has 57 original windows and 1,499
valid supervised frames (about 25 seconds). No test recordings were trained on.
Its own fitted-reference discrepancy is **0.35045 mm pooled** / 0.34837 mm
voice-macro. The previously quoted 0.373 mm was measured on a different 96-clip
cohort and is not the reference for these eight clips.

## Matched results

Both arms start at the same current v4 and retain all 331,047 parameters, original
waveform frontend, normalization, window/crop rules and lookahead. All backbone
weights train jointly. Each arm uses full-batch AdamW without weight decay,
3,000 updates, clipping 1.0, and cosine learning rate .0003→.00001. The only
intentional difference is production mixed loss versus pure normalized fitted-
coefficient MSE. Selection is lowest fitted-target gap RMS at the same checkpoints,
including epoch zero. This is diagnostic selection on the memorised data.

| Measurement | Initial v4 | Mixed loss, selected | Fitted-target MSE, selected |
| --- | ---: | ---: | ---: |
| Raw-teacher gap RMS, pooled mm | 1.75932 | 0.46053 | **0.39533** |
| Fitted-target gap RMS, pooled mm | 1.77170 | 0.45079 | **0.16314** |
| Raw-teacher gap RMS, voice-macro mm | 1.66589 | 0.45879 | **0.39280** |
| Normalized coefficient MAE | 0.30173 | 0.05932 | **0.03552** |
| Raw-teacher contact F1 | 0.88802 | **0.99901** | 0.97358 |
| Selected update | 0 | 800 | 3,000 |

The fitted reference itself has contact F1 0.98083 against raw teacher contact.
The fitted-MSE arm is 12.8% above its subset's reference and passes the predeclared
closeness screen (raw RMS ≤1.25× reference). Mixed loss is 31.4% above and fails
that screen. At update 3,000, mixed loss has 0.46910 mm raw / 0.46229 mm fitted-
target RMS and contact F1 1.00000; retain those final values as well as the selected
checkpoint. Pure fitted-MSE selected its final checkpoint. Both entire histories
are preserved. Training took about 21/19 seconds on M5 Max; no inference-speed
claim follows from those optimizer timings.

## Interpretation

The architecture can represent substantially more accurate outputs for a small
known set. A hard 2–3 mm error floor imposed by this file size is not supported.
This is consistent with, and extends, the earlier four-clip finding.

The matched objectives also expose a real trade-off: mixed loss favours raw
contact classification and timing/geometry terms as well as fitted coefficients.
Pure fitted-target MSE gives lower gap RMS but worse contact F1. It is not a
free production improvement. This experiment does not identify which individual
mixed-loss term causes the trade-off; that needs a separately controlled ablation.

Successful memorisation does not prove unlimited capacity, general phonetic
features, sufficient training diversity, or one unique explanation of the held-out
error. A network can use recording-specific cues. Generalization, optimization
and loss balance remain the leading questions to test, rather than treating the
model's small file size as a demonstrated accuracy limit. Broader capacity limits
can still exist even though eight recordings are learnable. The fitting reference
is not a rigorously optimal raw-gap lower bound; do not subtract RMS values as
independent error components.

A useful next controlled study would vary the coefficient/geometry/contact loss
balance on the full training corpus, using development selection and a fresh
confirmation set, retaining /m,b,p/ event metrics. Do not publish these memorised
weights or silently switch to MSE solely because its training RMS is lower.

## Native/export and source checks

Frozen trainer/protocol/data/audio/label hashes were verified. Both unchanged
waveform exports remain **2,251,827 bytes / 331,047 parameters**. Filters, mel bank,
normalization, interpolation and output channel order are bit-identical to v4.
Complete utterance inference on all eight original recordings reproduces pooled
training metrics to within 0.000002 mm.

The initial coefficient-parity check used 5e-5 tolerance. Mixed loss passed
(maximum difference 1.44e-5); pure fitted-MSE did not (2.44e-4). Preserve that
failure instead of loosening the threshold and reporting a pass. A supplementary
frontend check found that float32 ONNX convolution versus cached NumPy FFT mel
values differ by at most 6.14e-6 (mean 8.63e-8). Feeding that ONNX mel into the
same MLX network reproduces up to 2.42e-4 coefficient change for fitted-MSE, versus
1.45e-5 for mixed loss. Thus the trained fitted-MSE model amplifies small frontend
rounding differences. Aggregate lip-gap RMS remains unchanged at report precision,
so the memorisation conclusion survives native waveform inference; this is not
production browser-parity approval. No browser or iPhone performance was measured,
and no CDN/deployment action was taken.

Artifacts remain ignored under `.local/v4-memorisation-v1/`: manifest/protocol,
frozen source copies, full-batch arrays, complete trajectories, selected/last
weights, optimizer snapshots, native exports, numerical checks and learning
curves. The pre-existing model and deployed idle/WebGPU behaviour are unchanged.

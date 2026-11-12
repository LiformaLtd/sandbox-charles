# Teacher-feature distillation of current compact v4

Freeze this experiment before optimization and fresh confirmation scoring.
Published v4 scale-r3 remains unchanged throughout this research phase.

## Hypothesis and scope

The early 52-text multivoice pilot already tested raw feature-MSE supervision
and found no convincing additional gain. This experiment tests normalized
feature supervision on the much larger current corpus, not a first-ever trial.
Use the full teacher's last audio-encoder layer, already cached independently
of the recurrent geometry state, with the exact existing waveform/window/frame
alignment. No extra audio context, aligner or teacher encoder runs in the browser.

Initialize both arms from scale-r3 SHA
407739196cd98c89ec619b6af527f1021a3702cf8b1213c006f86408d1701137.
The student frontend, 96 channels, eight blocks, normalization, 52-output mapping
and export remain fixed. A training-only 96→768 affine head predicts normalized
teacher features from the student's final interpolated hidden states. It is
removed before export. Expected graph: 331,047 parameters / 2,251,827 bytes,
25 buffers / 1,492,236 explicit model GPU bytes / two compute pipelines.

## Matched optimization

Control and distillation instantiate the identical auxiliary head and use the
same seed-37 initialization, sampled minibatches and animation losses. Control
auxiliary coefficient is zero. Distillation adds mean 1−cosine similarity over
valid primary frames, coefficient 0.1 ramped linearly over the first 372 updates.
Normalize vectors using sqrt(sum(x²)+1e-6). Historical clean replay has no hint
loss; both arms retain its ordinary animation supervision. No other loss changes.

Use the existing expanded corpus: 3,328 primary originals / 3,840 gain-labelled
examples plus unchanged 549 clean historical replay clips. Reuse the audited
read-only mel/geometry caches. Build aligned f16 hint maps for the primary set
only; hash source labels and check target/window alignment. No teacher target
normalization is fitted using development or confirmation.

Use validated MLX float32 with TF32 disabled. Before training compare auxiliary
loss and gradients against CPU PyTorch, audit zero control hint gradients and
nonzero hint gradients reaching the backbone, and preserve an exported initial
student reference. Train both 3,720 updates, batch48 (six per eight voices) plus
16 replay, PCG64 seed1037, AdamW beta.9/.999 eps1e-8 decay.015 clip1, cosine LR
.00012→.000012. Evaluate at step0 and every124; select macro development voice
RMS+.25*(1−contact F1), including unchanged Easton guard. Initial checkpoint may
win. Save optimizer/RNG states and selected weights. No coefficient sweep or
extra seed selected by fresh test. This is one fixed pilot; a positive result
needs independent follow-up before release claims.

## Confirmation and runtime checks

Freeze 24 new disjoint texts (12 authored targeted +12 unseen ARCTIC), four known
training-held-out presets balanced by engine/gender, giving96 new recordings
and384 independently labelled original/half/headroom-boost/quarter cases. Exclude
all retained prior training, development and test transcripts, including scale.
Presets are previously studied, not new voices. Retain existing licensing and
source provenance. Freeze visual cases and evaluator before first test scores.

Primary comparison is distillation versus matched control on original macro
voice lip-gap RMS. Promising screen: >=3% improvement; no engine-average RMS or
contact-F1 regression; half/boost mean RMS no worse; well-timed /m b p/ total
not lower and no category loses more than2; historical/scale regression RMS at
most2% worse than current v4 and F1 at most.01 lower. Report every arm versus
current v4, all gain levels, opening errors and onset/release/closure results.
Keep a failed screen intact. Existing inspected scale confirmation is regression
only. No promise that feature matching improves lip sync.

Verify exact ONNX size/operator topology and absence of auxiliary weights,
native/WebGPU full streaming tracks and repeated-generation resets. Benchmark
three settled desktop WebKit runs after training/teacher activity ends. Measure
same active James idle page plus prerecorded short/long repeated STA using20ms
physical-footprint sampling. Report TTS separately/excluded; neither process
sampling nor an emulated iPhone user agent establishes physical-iPhone safety.
No model CDN upload or production deployment in this experiment.

# Frozen-v4 residual learning versus matched direct prediction

Design fixed2026-11-08 before scoring. No production model, defaults or idle
animation change. Generate all artifacts under ignored `.local/compact-residual-v1`.

Freeze published finetune-r4 v4, SHA256
`ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`.
Cache its96-channel learned audio features on the existing60fps timeline in
float32, preserving all60 frames per window. Both heads receive the same feature
vector and39 normalized v4 predictions. Feature normalization uses fixed training
batches only. No new encoder, phoneme model, history or lookahead is added.

Both arms add a135→96 GELU→39 pointwise head (16,839 trainable parameters).
Residual predicts a normalized coefficient correction added before clipping;
direct control predicts full normalized coefficients. Direct head initializes
39 positive/negative GELU pairs to reproduce v4, using GELU(x)−GELU(−x)=x;
residual head initializes its final layer to zero. Both first-layer initializations
are identical. The remaining18 hidden units have seed37 random inputs and zero
outgoing weights. This matches initial functions, head size, training data and
budget, not internal gradients or parameter values. The experiment compares
these two training parameterizations, not residual learning against all possible
direct architectures. Base weights are frozen in both arms.

Use the unchanged full-teacher coefficient/gap/velocity/contact/timing objective,
voice-balanced48 primary+16 replay windows, seed37 and7,440 scheduled AdamW
updates per arm. Learning rate .0003→.00003 cosine, betas.9/.999, epsilon1e-8,
weight decay.015, gradient norm limit1. Development selector is the existing
voice-macro gap RMS+.25*(1−contact F1), checked every248 updates; initialization
is eligible. Lock sources, model hash and source cache hashes before training.

The frozen base plus either head must export below2.35MB, versus2.251827MB v4.
This deliberately allows about3% more bytes for the shared-feature head; it is
not a claim of identical original file size. Both arms have the same parameter
count and almost identical runtime arithmetic. A research-only WebGPU runner
uses the existing audio/convolution kernels and one fused projection/head kernel.
No second session, waveform pass or model is loaded in the browser.

Before any quality scores, freeze24 new transcripts (12 targeted /m,b,p/vowels,
12 broad Arctic) disjoint from existing planned/training/test texts, with four
existing held-out Phonon/Supertonic voice presets. Generate and independently
teacher-label original/half/boost/quarter gain waveforms. Model selection uses
development only. Lock selected models before fresh scoring. Old confirmation
and historical sets remain regression evidence. Score complete streaming clips;
report original-gain primary training fit separately from test fit.

Require residual extra gap-RMS improvement≥3% over direct control, lower error
than v4, no per-engine RMS regression versus control, and macro/per-engine contact
F1 and /m,b,p/ hit counts no worse than both control and v4 (tolerance1e-6 for
F1). False-closure and missed-contact rates cannot increase over control beyond
1e-6; conditional onset/release MAE cannot worsen by more than.1ms. Retain the
existing actual-teacher event scorer including up to100ms anticipatory association.
Report those metrics even if a guard fails. Historical RMS≤v4*1.02 and F1≥v4−.01.

Verify frozen base/frontend tensors, initial function equality, native/ORT and
full streaming WebGPU parity, finite outputs, exact reset repeats and inactive
channels. Measure three fresh WebKit speed/footprint trials with animated James,
short/long/repeated prerecorded STA. Target block speed≤control*1.10 and≤v4*1.20,
same extra GPU allocations between arms, and<1MiB extra explicit buffers over v4.
Footprints exclude TTS, initial rig loading and browser parent; no physical-iPhone
or full-pipeline claim. No deployment/CDN publication without subsequent review.

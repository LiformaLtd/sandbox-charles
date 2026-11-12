# Independent representation-capacity experiment

Freeze before training and scoring. Published v4 coverage-r2 stays unchanged.

## Hypothesis and controlled change

The former width experiment replicated learned channels, leaving nearly
identical features. This test instead adds 32 independently initialized channels
to the existing 96-channel network, keeping the same 64-bin FFT/log-mel frontend,
eight residual blocks, dilations, one-second windows and 52-channel interface.
No pretrained speech encoder, phoneme aligner or text input is added at runtime.

Both arms initialize from the published coverage-r2 hash
`5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd`.
Width 96 is a matched additional-training control. Width 128 preserves all
original channels, independently initializes extra rows, and sets new-to-old
feedback and extra output weights to zero at initialization. Thus it initially
predicts the same function, but the additional features are not copied. All
parameters remain trainable. Verify initial prediction parity, low new/old filter
cosine similarity and gradient flow into new features after output weights open.

Use the same broader-speech primary corpus, labelled gain levels, common
old/new development cohort, historical replay, losses, normalization, seed 37,
learning rate and 24 × 31 maximum-update budget. This holds updates/data fixed,
not FLOPs or wall time: the wider model does more arithmetic. Select both
checkpoints with the unchanged development score and Easton guard; freeze hashes
before test scoring. Width 96 has 331,047 parameters; width 128 has 572,455.
Expected downloads are about 2.25 versus 3.22 MB. Do not assume the additional
capacity succeeds or add it to the public runner before compatibility checks.

## Fresh confirmation and regression checks

Freeze 24 additional original texts, balanced across the eight established
articulation categories. Synthesize them with the four training-held-out pilot
voice presets (two engines, both catalogue genders), giving 96 fresh recordings.
Generate original/half/headroom-boost/quarter variants with corresponding full
teacher labels. These are fresh texts/recordings, not previously unseen presets.
Exclude exact normalized overlaps with every earlier known compact corpus and
the new coverage study. No confirmation data selects checkpoints or widths.

Also report the already inspected coverage test cohort and historical gain/
Easton sets as regressions, not fresh evidence. Primary screen: fresh original
macro-voice gap RMS improves at least 5% versus the width-96 control, with no
engine-average or contact-F1 regression. Fresh half/boost mean RMS must not
regress; report quarter separately. Historical original RMS may regress at most
2% and contact F1 at most 0.01 versus published coverage-r2. Report opening and
closure timing separately, not just RMS. One-seed results remain exploratory;
confirm promising recipes independently before making broad population claims.

## Browser cost

If quality is promising, implement a research-only width-128 WebGPU runner with
the same two-kernel structure and stable GELU. Verify native/GPU agreement on
complete speech tracks, allocation counts and bytes, warm block speed and
repeat generation. Benchmark only after training ends. The estimated persistent
model GPU allocation rises from 1,492,236 to 2,496,652 bytes; that estimate is
not a total-page or physical-iPhone peak-memory measurement. Keep normal idle
animation. Do not silently fall back to CPU for the deployment target.

Artifacts stay ignored under `.local/compact-capacity-v1`, with source/data/model
locks and complete optimizer checkpoints. Runtime production files and released
weights are not changed by the research phase.

# Tiny causal temporal decoder

Freeze before training/scoring. Published v4 coverage-r2 is unchanged. This pilot
adds a short motion-feature history, not a heavy audio encoder or phoneme input.

Freeze v4 coverage-r2 backbone, including the current FFT/log-mel frontend,
96-channel temporal encoder and original output projection, SHA
`5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd`.
A residual decoder takes eight successive 96-dimensional resampled backbone
features (current plus seven past frames, about 117 ms history), projects to
16 GELU units, then to 39 active coefficient corrections. Add scaled corrections
to the original clipped prediction. Initialize correction output to zero so
both arms initially reproduce v4. No corrected pose is fed back autoregressively.

Temporal arm receives real causal feature history. Matched control has identical
weights, operators and parameter budget but repeats current-frame features in
all eight slots. Both use the same backbone/feature normalization, decoder seed
37, frozen primary/replay data, sampled windows and 24 x 31 AdamW updates,
LR 0.0003 cosine to 0.00003, weight decay .015, existing geometry/contact/timing
losses. Only decoder parameters train. Sample 48 balanced primary plus 16 clean
historical windows per update. Normalize features from training frames only.
Construct past context on complete clip tracks before drawing windows; never
reset history at a training block boundary or carry it between recordings.

Select checkpoints on the unchanged macro-voice gap RMS + .25*(1-contact F1)
development criterion, including Easton guard. Freeze exports before scoring.
Test against published v4 and matched control, not just initialization. This
matched decoder control is different from the prior full-backbone fine-tune.

Fresh confirmation: 24 unused texts, four known training-held-out voice presets,
96 recordings, original/half/headroom-boost/quarter teacher-labelled variants.
Exclude previous known transcript overlap, including state/capacity cohorts.
Primary screen: >=5% fresh original macro-voice RMS improvement versus control,
no engine-average or contact-F1 regression, no half/boost mean RMS regression.
Historical and coverage original RMS <=2% worse and F1 <=.01 worse than v4.
Report quarter, opening error and /m b p/ event hits/onset/release separately.
One seed is exploratory; no release claims without independent confirmation.

Export explicit normalized history [1,7,96], first/count inputs and next history.
Update history only for emitted valid frames. Reset to zero between utterances;
exclude initial all-padding blocks. Test chunk invariance, causality, short tails,
reset/repeated generation and native/GPU parity. A research-only third WebGPU
pipeline adds the decoder after original projection; reuse buffers, including
fixed scratch/readback, and keep original idle animation. Do not use ORT WebGPU
or a CPU fallback for the temporal runner. Target about 52 KB extra learned
weights, 2.30 MB total download and only a few KB of history/scratch GPU storage.

Benchmark both arms and v4 after training: warm blocks, startup, explicit GPU
buffers, long/repeated streams. Report model buffers separately from full-page
peak memory. Desktop WebKit is not a physical-iPhone memory guarantee. Keep all
weights/audio/checkpoints/locks under ignored `.local/compact-temporal-v1`.

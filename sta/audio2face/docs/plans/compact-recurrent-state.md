# Jointly trained small recurrent state pilot

Protocol fixed before training, 8 November 2026. Current v4 finetune-r4 is the
baseline (SHA ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da).
The teacher-history diagnostic justifies testing recent history, not a promised
large gain. No automatic publication follows this experiment.

Both arms retain the exact v4 waveform/log-mel frontend, 96-channel backbone,
one-second input windows, 500 ms hops and emitted crops. No extra lookahead.
Add a 32-unit frame-rate GRU on tanh(encoded features / 4), then a zero-initialized
39-channel linear residual in the original standardized coefficient units.
All original backbone and new decoder parameters train jointly. In the recurrent
arm, hidden state persists over emitted frames/blocks. In the control the same
cell resets its input state at every frame; recurrent matrices are allocated but
receive no gradient from zero previous state. Parameter counts are identical;
effective temporal capacity intentionally differs. Both start exactly at v4.

Use only existing scale-study old/new training and historical replay corpora.
Precompute a shared deterministic chronological stream schedule, seed 1037:
20 slots, two per eight training voices plus four replay slots. Each update
consumes up to four consecutive original blocks per slot. Reset when a new clip
starts; never cross clips, advance through padded frames, or duplicate overlap
frames into history. Detach carried state at each four-block truncated-BPTT
boundary. States generated before optimizer updates are ordinary truncated-BPTT
carry, not re-encoded after every update. Both arms share all sampler indices,
resets, optimizer, objective and budget.

Budget: 1,240 updates, AdamW 0.00012→0.000012 cosine, weight decay .015,
gradient norm clipping 1.0, seed 37, existing coefficient/gap/contact/timing loss.
This is a bounded architectural pilot, shorter in updates than the prior 3,720
fine-tuning trials; sequence exposure is reported separately. Evaluate complete
ordered development clips every 124 updates. Retain epoch-zero as an eligible
checkpoint. Select voice-macro gap RMS plus .25*(1-contact F1), in centimetres,
on common development plus historical development only. No test selection.

Freeze 24 fresh transcripts (12 targeted, 12 broad), four held-out voice presets,
96 recordings, and original/half/boost/quarter waveform variants before scoring.
Exclude all prior training and confirmation transcripts; use pinned Phonon and
Supertonic synthesis, alignment and full-teacher labels. Report nine-gap RMS,
contact F1, velocity error, /m,b,p/ closure success with the existing 0–100 ms
anticipatory matching, per-engine and per-gain results. Existing surface/feature
confirmation and historical test sets are regression evidence, not fresh tests.
Surface error can be measured on the existing labeled surface cohort separately.

Primary architecture screen: at least 3% gap RMS improvement over matched control,
no engine worse, no original contact-F1 loss over .002, no total well-timed-closure
loss, and no phone losing over two hits. Historical/feature regression gap must
stay within 2% and F1 within .01 of v4. Report all metrics even if the screen fails.

Export explicit state/crop inputs. Validate native training/export parity,
causality, cross-block carry, short tails, resets and complete browser tracks.
Compare v4/control/recurrent using production frontend/convolution WebGPU code
plus a research-only fixed-buffer GRU pipeline. Measure warm block times and
renderer+GPU simultaneous physical footprint at 20 ms sampling on desktop WebKit,
with James continuously animating idle, short/long/repeated prerecorded clips.
Include startup and state/buffer bytes; exclude TTS and clearly state that these
are not full-pipeline physical-iPhone guarantees. Keep model/file/state overhead
small; report speed/memory increases honestly. Do not change default demo or CDN.

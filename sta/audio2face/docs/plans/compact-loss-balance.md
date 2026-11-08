# Full-corpus fitted-coefficient loss rebalancing

Predeclared 8 November 2026, before training or fresh confirmation scoring.
Start both arms from released v4 finetune-r4, SHA
`ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`.
No architecture/frontend/normalization/window/lookahead/runtime changes.

Memorisation supports learnability on tiny known sets, not a production MSE
switch: fitted-target MSE reduced RMS but lost contact F1. Test one fixed change,
coefficient SmoothL1 weight 1→4, retaining all other terms exactly: raw gap1.2,
velocity .12, contact BCE .22, near-contact timing .45. This is not pure MSE,
a sweep or a separately identified contact-term ablation. The factor4 is chosen
as an explicit hypothesis, not an empirically optimal weight. No tuning after
confirmation results. Fourfold coefficient emphasis changes the gradient norm;
retain identical optimizer/clipping rather than claiming equal update norms.

Full existing scale primary corpus (old-train + new-train) and original/extra
historical replay, without filtering difficult examples or adding new training
recordings. Both arms use 3,720 updates, six primary windows per eight voices plus
16 replay windows per batch, seed37, identical sampled indices, AdamW .00012→
.000012 cosine, weight decay .015 and norm clipping1.0. Evaluate every124 updates
on common-dev plus replay-dev with the same voice-macro gap RMS + .25*(1-contact F1)
selector in centimetres. Include epoch-zero; save complete optimizer/RNG/best state.

Freeze 24 new texts (12 targeted and 12 broad), four held-out presets and 96
recordings before training. Separately full-teacher-label original/half/boost/
quarter waveforms (384 cases), excluding all known prior training/confirmation
texts, including recurrent confirmation. Freeze both checkpoints before scoring.
Primary: voice-macro original-level nine-gap RMS. Also report contact F1,
/m,b,p/ well-timed closure hits with existing 0–100ms lead allowance, velocity,
per-engine and per-gain results. Prior feature/surface/recurrent and historical
sets are regression evidence, not fresh confirmation.

Promising screen: at least 3% extra RMS gain over matched control; neither engine
worse; contact F1 no more than .002 lower; total well-timed closure hits not lower;
no phone loses more than2 hits. Historical/feature RMS within2% and F1 within.01
of released v4. Report negative results and any trade-offs. No automatic promotion.

Export exact unchanged one-input waveform ONNX graph. Require native/MLX and
complete-stream browser parity; compare original v4 and both exports for WebGPU
block processing, allocations and renderer+GPU physical footprint with James
still animating idle. Use desktop WebKit, three fresh-browser memory trials,
short/long/repeated prerecorded STA, 20ms sampling, no TTS. This does not establish
physical-iPhone full-pipeline memory. No CDN/deployment change is authorized by
this diagnostic alone.

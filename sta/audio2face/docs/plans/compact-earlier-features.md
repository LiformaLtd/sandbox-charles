# Earlier audio features: matched residual-head ablation

Design fixed2026-11-09 before any new training or scores. Keep published v4,
WebGPU default and continuous idle animation unchanged. Generated artifacts stay
ignored under `.local/compact-early-v1`; no publication or deployment in this pilot.

Freeze all published finetune-r4 weights (SHA256
`ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`).
Late control uses its96-channel final learned audio features, as in the preceding
residual pilot. Early treatment uses the96 channels immediately after its first
5-tap audio convolution and GELU, before all eight dilated residual blocks. This
is an earlier learned representation of the shared log-mel frontend, not raw
log-mel input, an added encoder or a new temporal decoder. Both branches retain
the frozen base predictions, which still use all original context/lookahead.

Resample both feature sources onto the same60fps timeline. Both heads receive
96 source features and the identical39 normalized base predictions. Freeze all
cached features/predictions in float32 and use exactly the same base prediction
cache for both arms. Source scales use the same32 fixed training batches, with
independent channel RMS normalization appropriate to each source. No test stats.

Both arms use the same135→96 GELU→39 pointwise residual head,16,839 trainable
parameters, identical seeded first-layer weights and zero outgoing weights.
Thus both initially reproduce v4. Only the source of the96 input features and
its training-derived normalization differs. No additional frames, history,
phoneme input, recurrent state, loss term or lookahead. Frozen normalized base
predictions are added to head output before final scale/mean and clipping.

Use the previous pilot's identical seed37,7,440 scheduled updates,48 voice-balanced
primary+16 replay windows, original geometry/coefficient/contact/timing objective,
AdamW .0003→.00003 cosine, betas.9/.999,epsilon1e-8,decay.015,grad limit1. Select
every248 updates with voice-macro dev gap RMS+.25*(1−F1); initialization eligible.
Lock all sources, input model, original arrays, new caches and normalization
before training. Train both arms from scratch; do not select on any test score.

Freeze24 fresh transcripts, including12 targeted /m,b,p/vowel prompts and12
Arctic sentences, disjoint from all prior planned/labelled texts. Use four voice
presets outside training across Phonon/Supertonic,96 original recordings and384
separately teacher-labelled gain cases. Complete corpus audit and model selection
locks before scoring. Earlier test sets are inspected regression evidence only.

Require early extra RMS gain≥3% over late, lower RMS than v4, no per-engine RMS
regression versus late, no macro/per-engine contact-F1 loss or per-phone /m,b,p/
hit loss versus both late and v4. F1 tolerance1e-6. False closure/missed-contact
rates cannot worsen versus late beyond1e-6; conditional onset/release MAE may
worsen at most.1ms. Keep the existing actual-teacher event scorer including up
to100ms anticipatory association. Historical gap≤v4*1.02,F1≥v4−.01.

Export both whole waveform models below2.35MB, with the original331,047 parameters
frozen plus equal16,839-parameter heads. Use one research-only WebGPU runner with
identical allocation/dispatch topology for both sources. Retain the encoder
activation in one extra buffer, with no extra audio/encoder pass or session.
Check fixed-base constants, cache alignment, initial outputs/losses, full native
and GPU waveform parity, inactive channels and repeated resets. Three fresh
desktop WebKit speed/footprint trials run after all GPU training/labelling.
Require early speed≤late*1.10 and≤v4*1.25, plus<1MiB extra explicit GPU buffers
over v4. Keep James animating. Footprints exclude TTS, initial rig load and browser
parent; do not claim physical-iPhone or full-pipeline memory. A negative pilot
does not rule out raw spectral features or temporal correction heads.

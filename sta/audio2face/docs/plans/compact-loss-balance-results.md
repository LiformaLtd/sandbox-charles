# Full-corpus coefficient-loss balance: matched pilot

Completed 8 November 2026. **Do not adopt the fixed 4× coefficient weight.** It
slightly worsened fresh original-level gap RMS and contact F1 versus matched
fine-tuning, despite some improvements on quieter speech and total closure hits.
Published v4 finetune-r4 is unchanged; no CDN upload or deployment was performed.

[Predeclared protocol](compact-loss-balance.md) ·
[Reproduction](../../scripts/research/loss_balance/README.md)

## Matched experiment

Both arms initialize from published v4, SHA
`ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`.
The only intended change is the normalized fitted-coefficient SmoothL1 weight,
1→4. Gap1.2, velocity.12, contact BCE.22 and near-contact timing.45 remain fixed.
This is a single predeclared ratio, not pure MSE, a hyperparameter sweep or a
separate contact-term ablation. It was chosen as a hypothesis, not an optimum.

The complete existing primary training corpus contains 3,840 labelled examples
(old-train768 + new-train3,072), plus original/extra historical replay. No new
training examples or error-based filtering were introduced. Both arms use the
same 3,720 updates, six primary windows per eight training voices plus sixteen
replay windows, RNG seed37/1037, AdamW .00012→.000012 cosine, weight decay.015,
and gradient clipping1.0. The final sampler SHA is identical:
`60025a93a676c1dc18ce066fb3f0160484b7e3142796d7d1ab66cc2efb4258b6`.
All feature arrays and primary sources were hashed before training.

The same complete common-dev/replay-dev voice-macro gap/contact selector, including
epoch-zero, selected update2,976 for control and1,116 for rebalancing. Initial
development scores are identical. Both finished the full budget; different selected
steps are a consequence of the same selector, not a different training budget.
The rebalanced objective clipped gradients on100% of sampled updates, versus
68.63% for control; mean preclip norms were2.621 versus1.150. Identical optimizer
settings do not mean identical effective gradient magnitudes. This is an observed
effect of changing the objective, not a separately isolated causal mechanism.

## Fresh confirmation

Before training, 24 new transcripts were frozen: twelve targeted /m,b,p/ and
vowel cases plus twelve broad CMU Arctic sentences, each in held-out voice presets
Phonon Freya/Hugh and Supertonic F5/M5. Exact prior training/confirmation overlap
was excluded, including the recurrent study. Ninety-six original recordings plus
separately full-teacher-labelled half/boost/quarter variants give384 cases.
One unsupported dictionary word was replaced before protocol freeze or synthesis;
no prompts were changed in response to any model errors. Checkpoints were locked
before scoring. Absolute RMS differs from earlier studies because this is a new
cohort; always compare models on the same clips.

| Original-level metric | Published v4 | Matched control | Coefficient4× |
| --- | ---: | ---: | ---: |
| Voice-macro lip-gap RMS, mm | 2.28384 | **2.24551** | 2.25118 |
| Contact F1 | 0.80041 | **0.80098** | 0.79670 |
| Well-timed /m/ hits | 29 | 25 | 23 |
| Well-timed /b/ hits | 72 | 71 | 74 |
| Well-timed /p/ hits | 58 | 57 | 58 |
| Total well-timed hits | 159 | 153 | 155 |
| Clip-mean coefficient MAE | 0.03156 | **0.03109** | 0.03122 |
| Clip-mean gap velocity MAE, mm/frame | 0.43552 | **0.43386** | 0.43911 |

The candidate is **0.253% worse in gap RMS than matched control**, with contact
F1 lower by0.00429. It is1.430% better than published v4 in RMS, but control
improves more; do not attribute ordinary fine-tuning gains to the new balance.
Although total well-timed hits rise153→155 versus control, /m/ loses two and both
fine-tunes have fewer total hits than released v4. Closure association retains
the existing0–100ms anticipatory allowance; total counts are not a substitute
for per-phone, continuous geometry and frame-contact measurements.

The predeclared3% extra-RMS screen, no-engine-regression guard and contact-F1
within.002 guard all fail. The total-hit, per-phone and historical/feature guards
pass. A supplementary2,000-replicate paired bootstrap over24 transcript clusters,
retaining all four voices, gives95% extra-gain interval−1.552% to+1.141% versus
control. This spans zero, is conditional on these voices/one training seed, and
was not used for model selection. It does not establish a significant degradation
or any useful improvement, and the practical predeclared screen still fails.

| Engine gap RMS, mm | v4 | Control | Coefficient4× |
| --- | ---: | ---: | ---: |
| Phonon | 2.14729 | **2.11467** | 2.12774 |
| Supertonic | 2.42039 | 2.37634 | **2.37462** |

Phonon worsens0.618% versus control; Supertonic improves0.073%.

| Waveform condition gap RMS, mm | v4 | Control | Coefficient4× |
| --- | ---: | ---: | ---: |
| Original | 2.28384 | **2.24551** | 2.25118 |
| Half | 2.38482 | 2.35689 | **2.33307** |
| Boost | 2.40204 | **2.35347** | 2.35827 |
| Quarter | 2.81208 | 2.79710 | **2.72337** |

Half/quarter volume benefit (1.01%/2.64% versus control) does not retroactively
replace the original-level primary endpoint or its contact guards. Keep the
positive stratified result without declaring an overall win.

## Prior-cohort regression evidence

These are previously inspected cohorts, not fresh confirmation:

| Gap RMS, mm | v4 | Control | Coefficient4× |
| --- | ---: | ---: | ---: |
| Surface confirmation | 2.46046 | **2.41529** | 2.43880 |
| Feature confirmation | 2.42607 | **2.38722** | 2.40565 |
| Historical | 2.54635 | 2.49198 | **2.47303** |
| Recent recurrent confirmation | 2.46400 | **2.41571** | 2.44600 |

Supplementary raw dense-mouth 3D RMS on the prior surface cohort is2.89763 mm for
v4,2.84445 for control and2.86464 for the candidate. It worsens0.710% versus
control. This does not support a hidden surface-detail benefit. All regression
scores and per-engine/contact details remain in ignored JSON reports.

## Browser speed, memory and parity

Architecture, one-second waveform frontend, normalization, interpolation,
lookahead and one-input waveform ONNX operator graph are unchanged. All models
have **331,047 parameters /2,251,827 bytes** and use the unchanged production
custom WebGPU runner. No new decoder, state or browser dependency was added.

On Apple M5 Max /64GiB, Playwright1.65.0 /desktop WebKit27.2 revision2388:

| Measurement | v4 | Control | Coefficient4× |
| --- | ---: | ---: | ---: |
| Warm long-clip block median, ms | 1.02 | 1.01 | 1.02 |
| Nominal500ms /processing factor | 490× | 495× | 490× |
| Explicit GPU buffer bytes | 1,492,236 | 1,492,236 | 1,492,236 |
| Cached-local session creation median, ms | 9.44 | 9.32 | 9.66 |
| Renderer+GPU peak range, MiB | 522.6–529.6 | 525.0–532.8 | 528.5–530.3 |

Timing is the median of three trials' medians on complete27.52s Phonon and33.44s
Supertonic tracks. It includes frontend/inference/readback, excludes TTS, loading
and rig draw work; tiny differences do not establish a speed benefit. This factor
is not end-to-end generation throughput. All eleven short/gain/packed/tail fixtures
plus a reset repeat were checked in three full browser runs per model.

Nine fresh-browser physical-footprint trials sampled simultaneous WebContent+GPU
sums every20ms, from loaded animated James idle through model startup, a short
Phonon clip, a27.52s clip and three long repeats. A research-only pose hook applies
coefficients to James; idle head/blink and render drawing continue. All trials
had advancing draw counters and no page errors. Sampling can miss short peaks;
initial rig loading, TTS, browser-parent/network processes and physical iPhone
are excluded. Accelerated prerecorded STA does not constitute a paced full
TTS+STA pipeline. Overlapping peak variation is not a memory reduction.

Control loss values AND parameter gradients are byte-identical to the original
implementation. Frozen data/source hashes, sampler reconstruction, final RNG and
checkpoints passed audit. Original export graph and constant tensors are identical;
complete-stream MLX/native coefficient deltas are below8.95e-7. Three full WebGPU
trials passed the5e-5 threshold with no nonfinite/zeroed frames. Reset repeats are
byte-identical. Sixteen synchronized teacher/v4/control/candidate review cases
passed audio/play/seek checks; screenshots/curves are qualitative inspection,
not a separate blinded perceptual evaluation.

## Decision and limits

Retain published v4 finetune-r4. The matched control is also not automatically
promoted merely for lower RMS: total closure hits declined on this fresh cohort.
Do not silently substitute either fine-tune into the demo.

Tiny-set memorisation showed that low training error is possible; a fixed4×
coefficient weight did not reproduce that benefit on held-out full-corpus speech.
It did not improve held-out coefficient MAE versus control either. These results
reject this particular balance under this recipe, not every possible loss-balancing
method or a general capacity hypothesis. Gradient rescaling/clipping and target/
geometry/contact trade-offs remain possible explanations, not isolated causes.
No post-confirmation weight search, additional seed selection, CDN upload or
Vercel deployment was performed. Reproduction scripts and full local artifacts
remain under `.local/compact-loss-v1`, with model/audio arrays excluded from Git.

# Small jointly trained recurrent state: matched pilot results

Completed 8 November 2026 on Apple M5 Max / 64 GiB. **Do not replace published
v4 with this recurrent model.** Recurrence gave a measurable but tiny 0.177% extra
gap improvement over matched stateless fine-tuning, and slightly worsened Phonon.
The predeclared 3% architecture screen and no-engine-regression guard failed.

[Protocol](compact-recurrent-state.md) ·
[Reproduction](../../scripts/research/recurrent_state/README.md)

## Architecture and matched training

Current v4 finetune-r4 was pinned at
`ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`.
The waveform filters, mel bank, normalization, 60 fps interpolation and output
channel order are bit-identical in both exports. One-second audio inputs,
500 ms hops and original emitted crops are unchanged: no extra lookahead.

A 32-unit GRU consumes `tanh(encoded / 4)` and supplies a zero-initialized linear
residual in standardized blendshape units. All backbone and decoder weights train
jointly. The stateless control has exactly the same cell/head and parameter
count, but resets the cell's previous state at every emitted frame. Its recurrent
matrix is allocated but dormant; effective access to temporal information is the
intentional difference. Both models start exactly at v4.

Both arms used the same frozen seed-1037 chronological schedule, two slots per
eight training voices plus four historical replay slots. A step consumes up to
four consecutive original blocks per slot; hidden state carries between steps,
is detached at each truncated-BPTT boundary, and resets between recordings.
Padding and overlap are never added to state. This pilot uses 1,240 AdamW updates,
83,731 nonpadding block exposures and 11,329 recording starts per arm. It is a
shorter update budget than prior 3,720-step fine-tunes, not an exhaustive search
of all recurrent training recipes. Both arms selected update 1,116 using the
same complete-clip development gap/contact selector. Epoch zero was eligible.
Training took about 79 seconds for control and 83 for recurrence on this machine;
these times are not browser performance measurements.

## Fresh lip-sync confirmation

Before scoring, 24 new transcripts were frozen: twelve /m,b,p/ and vowel tests,
twelve broad CMU Arctic sentences, each in four held-out voice presets (Phonon
Freya/Hugh, Supertonic F5/M5). Exact prior training/confirmation text overlap was
excluded. Ninety-six original recordings plus separately teacher-labeled half,
boost and quarter variants give 384 cases. Checkpoints were locked before scoring.

Original-level results, voice-macro gap/contact metrics:

| Metric | Published v4 | Matched stateless | Recurrent |
| --- | ---: | ---: | ---: |
| Lip-gap RMS, mm | 2.46400 | 2.43582 | 2.43150 |
| Contact F1 | 0.79246 | 0.79541 | 0.79571 |
| Well-timed /m/ hits | 50 | 51 | 52 |
| Well-timed /b/ hits | 55 | 58 | 58 |
| Well-timed /p/ hits | 47 | 49 | 49 |
| Total well-timed hits | 152 | 158 | 159 |
| Clip-mean gap velocity MAE, mm/frame | 0.44851 | 0.44701 | 0.44517 |

Closure scoring preserves the established 0–100 ms anticipatory association;
these are matched events, not requiring closure at the phoneme timestamp itself.
The recurrent gap improvement is **1.319% versus v4**, but only **0.177% versus
control**. Most of the baseline-relative gain comes from further fine-tuning.

| Engine, gap RMS mm | v4 | Control | Recurrent | Recurrent versus control |
| --- | ---: | ---: | ---: | ---: |
| Phonon | 2.34818 | 2.32110 | 2.32250 | 0.060% worse |
| Supertonic | 2.57981 | 2.55054 | 2.54050 | 0.394% better |

| Waveform condition, gap RMS mm | v4 | Control | Recurrent |
| --- | ---: | ---: | ---: |
| Original | 2.46400 | 2.43582 | 2.43150 |
| Half | 2.65590 | 2.63077 | 2.62888 |
| Boost | 2.49386 | 2.46562 | 2.46076 |
| Quarter | 3.07187 | 3.03187 | 3.03245 |

A supplementary 2,000-replicate paired bootstrap over the 24 transcript clusters,
retaining all four voices, gives a 95% interval of 0.117–0.245% for the extra gap
improvement. This is conditional on this cohort, chosen voices and one training
seed; it does not estimate training-seed uncertainty or prove a useful visual gain.
It was computed after primary scoring and was not used for checkpoint selection.

Historical and prior feature confirmation checks passed: recurrent gap RMS was
2.48448 and 2.41185 mm, versus v4 2.54635 and 2.42607. On the previously inspected
surface cohort, gap RMS was 2.44112 (control 2.44217, v4 2.46046). Supplementary
raw dense mouth 3D RMS was 2.85060 mm (control 2.85180, v4 2.89763): only 0.042%
extra dense-surface improvement over control. That is regression evidence, not a
fresh dense-surface test. Both older cohorts were excluded from training here.

A deterministic sixteen-case review compares teacher-fitted James, v4, control
and recurrence with a shared audio clock and gap curves. All sixteen audio/play/
seek cases passed browser checks. Inspected screenshots show very similar
control/recurrent shapes; the numerical result should not be marketed as a
substantial visual improvement. Review cases were selected by voice/category,
not by measured gain, after scoring; they are qualitative inspection only.

## Browser speed and footprint

Research-only WebGPU code retains the production FFT, convolution and projection,
adding one fixed-buffer GRU pipeline. It applies the residual before coefficient
clamping, matching training. Only emitted frames update state; reset between
utterances. Download sizes below are the actual waveform ONNX exports, including
unrolled crop/state graph overhead, not just neural weights.

| Metric | v4 | Control | Recurrent |
| --- | ---: | ---: | ---: |
| Parameters | 331,047 | 344,750 | 344,750 |
| Download bytes | 2,251,827 | 2,557,702 | 2,553,165 |
| Explicit GPU buffer bytes | 1,492,236 | 1,547,196 | 1,547,196 |
| Warm long-clip block median, ms | 1.05 | 1.23 | 1.23 |
| Nominal 500 ms block / processing real-time factor | 476× | 407× | 407× |
| Cached-local session creation median, ms | 9.74 | 9.76 | 9.86 |
| Renderer + GPU peak range, MiB | 522.9–534.1 | 530.9–536.2 | 527.0–533.9 |

Block timing is the median of three trials' medians across both 27.52-second
Phonon and 33.44-second Supertonic tracks. Recurrent trial medians were
1.25/1.23/1.19 ms, versus v4 1.07/0.97/1.05. This is about 17% more STA block
processing time than v4, still fast; it excludes TTS, model loading and rig draw
work. Short clips, four waveform gains and a 112.5 ms tail were also checked.
Do not treat this isolated STA factor as end-to-end generation throughput.

The cell adds 13,703 parameters, 128 bytes of persistent state and 54,960 explicit
GPU buffer bytes including weights/uniforms/state. The larger ONNX file also
contains unrolled graph nodes; that graph is not instantiated by the custom
WebGPU runner. A fresh-start browser trial per model, repeated three times,
measured simultaneous WebContent+GPU physical footprint every 20 ms. Trials run
from the loaded animated idle page through model startup, a 3.12-second clip,
a 27.52-second clip and three long repeats. Coefficients are passed to James
through a research-only page hook; head/blink/renderer animation continues.
All nine trials retained advancing draw counters and no page errors.

Peak variation overlaps; these results show no substantial memory saving, and
cannot resolve a small architecture-dependent physical-footprint change. Initial
rig-loading peaks, browser-parent/network processes and TTS are excluded. The
prerecorded inference loop is accelerated, not paced live speech. Sampling can
miss sub-20-ms peaks. This is desktop WebKit 27.2 / Playwright 1.65.0 on M5 Max,
not a physical-iPhone or complete TTS+STA memory guarantee. CPU ONNX was used
for reference scoring; the deployed WebGPU default and idle animation are unchanged.

## Validation and decision

Eight model/runtime unit checks passed, including zero-head v4 equality, padding,
frame-state independence of control, causality across blocks, partitioned carry,
recording boundaries, schedule continuity and all frozen-source/data hashes.
Export/native coefficient deltas were below 3.2e-7; state deltas below 1.9e-7.
Across three WebGPU trials and all complete clips, maximum coefficient delta
was below 7.1e-7. There were no zeroed/nonfinite frames; reset repeats were
byte-identical. Empty native crops leave state exactly unchanged.

An early export command stopped because recurrent training had not completed;
the final exports used the completed locked checkpoints, with no training rerun
or weight changes. All original and final logs are retained.

The teacher-history diagnostic demonstrated real missing information, but that
does not establish how much this compact student can recover from its current
features or under this pilot budget. This particular jointly trained state is
cheap and valid in the browser, yet does not deliver the larger quality gain we
wanted. Keep published v4 finetune-r4; no CDN upload or deployment was made.
Do not conclude that all possible recurrent architectures are exhausted.

# Tiny temporal decoder: results

**Keep published v4 coverage-r2.** This short-history, frozen-backbone pilot gained
only 0.52% fresh lip-gap accuracy, while well-timed closure hits fell. It failed
the frozen 5% quality target. Browser cost stayed small in absolute terms, but
was higher than v4. No model was published or deployed from this experiment.

## Controlled design

The original 64-bin FFT/log-mel frontend, 96-channel encoder and projection were
frozen at published v4. The residual decoder receives eight consecutive encoded
feature frames (current plus seven prior, approximately 117 ms history), passes
through 16 GELU units and predicts 39 scaled coefficient corrections. Its output
weights start at zero. The history contains normalized features, not previous
corrected poses; this is a finite-history causal decoder, not an autoregressive
GRU, and adds no lookahead to the existing audio-window schedule.

Matched control uses the same 12,967 decoder parameters and operators but repeats
current-frame features in all eight positions. Both arms use identical training-
only normalization, seed 37, source data/replay, sampled windows, losses and 744
AdamW updates. The head alone trains. Context was stitched across block boundaries
on whole clip tracks before sampling; it never crossed recordings. This control
is different from earlier studies' full-backbone fine-tune.

Both completed the fixed budget. Development selection retained epoch 0 for the
control: none of its trained checkpoints beat zero correction. The temporal arm
selected epoch 13. Exports were locked before fresh scoring. Backbone tensors
were bit-identical to v4, sampler/config audits passed, and native streaming
matched cached training features within 5.07e-7. Epoch-zero control reproduces
v4 exactly on the audited track; its fresh metrics also match v4.

## Fresh quality

24 new texts × four known training-held-out presets gave 96 new recordings,
48 per engine; original/half/headroom-boost/quarter levels gave 384 individually
teacher-labelled cases. Exact normalized overlap with known earlier corpora,
including state/capacity confirmation, was excluded. Text/audio are fresh;
presets are not previously unseen speakers. Checkpoints did not use this cohort.
The evaluator source was frozen after training but before confirmation scoring.
RMS is macro-voice James lip-gap error in millimetres over nine gap measurements.

| Condition | Published v4 | Matched control | Temporal |
|---|---:|---:|---:|
| original | 2.6922 | 2.6922 | 2.6781 |
| half | 2.9310 | 2.9310 | 2.9135 |
| boost | 2.6936 | 2.6936 | 2.6881 |
| quarter | 3.2749 | 3.2749 | 3.2596 |

Temporal original RMS improved **0.52%** versus control/v4, about **0.014 mm**.
Phonon improved 0.82%, Supertonic 0.29%. Half/boost mean improved 0.41%. There
is no independent-seed confirmation of a useful perceptual gain.

| Original measure | Published v4 | Matched control | Temporal |
|---|---:|---:|---:|
| Contact F1 | 0.770227 | 0.770227 | 0.770730 |
| Opening-only RMS (mm) | 3.0900 | 3.0900 | 3.0636 |

Overall contact F1 improved slightly, but short-event reliability did not.
Phoneme labels only associate actual teacher closures (100 ms lead / 35 ms
trail); geometry event times are never shifted to acoustic onset. Timing MAE
is conditional on overlap and must be read alongside hit counts.

| Phone | Teacher events | v4/control → temporal overlap hits | Well-timed hits | Onset MAE ms | Release MAE ms |
|---|---:|---:|---:|---:|---:|
| /m/ | 87 | 65 → 64 | 58 → 54 | 15.90 → 16.15 | 9.74 → 9.90 |
| /b/ | 98 | 84 → 83 | 75 → 74 | 15.48 → 16.27 | 14.88 → 14.86 |
| /p/ | 83 | 64 → 62 | 53 → 52 | 19.53 → 18.01 | 13.28 → 14.25 |

Well-timed hits fell in all three categories, despite slightly higher frame-level
contact F1. Holding the right pose for many frames is not equivalent to capturing
more transient closures. This is an additional reason not to promote the pilot.

Previously inspected regression cohorts, original RMS:

| Cohort | Published v4 | Matched control | Temporal |
|---|---:|---:|---:|
| historical | 2.7546 | 2.7546 | 2.7317 |
| coverage_regression | 2.6031 | 2.6031 | 2.5874 |
| capacity_regression | 2.6024 | 2.6024 | 2.5880 |

Historical/coverage RMS and F1 guards passed. These are regression evidence,
not fresh confirmation; improvements there do not overcome the primary failure.

## Browser speed and explicit memory

The prototype adds one WebGPU pipeline after original convolution/projection.
It reuses the FFT frontend and shaders and holds fixed history/output buffers.
Only emitted valid frames advance history; an utterance reset clears it. All
production files, published weights and normal idle animation remain unchanged.

| Measure | v4 | Matched control | Temporal |
|---|---:|---:|---:|
| Learned parameters | 331,047 | 344,014 | 344,014 |
| ONNX download bytes | 2,251,827 | 2,310,854 | 2,310,096 |
| Warm long-track block median | 1.01 ms | 1.42 ms | 1.48 ms |
| Six per-track median range | 0.96–1.08 ms | 1.40–1.44 ms | 1.42–1.50 ms |
| Approx. STA-only realtime, 500 ms emission | 495× | 352× | 338× |
| Explicit model GPU buffers | 25 | 28 | 28 |
| Explicit model GPU bytes | 1,492,236 | 1,560,060 | 1,560,060 |
| Compute pipelines | 2 | 3 | 3 |
| Cached-local session creation median | 9.32 ms | 10.26 ms | 10.40 ms |

The candidate adds 58,269 file bytes (2.6%), 67,824 explicit GPU bytes (4.5%) and
2,688 bytes of normalized history. The decoder also uses 3,136 bytes of WGSL
workgroup scratch; that is not a `createBuffer` allocation. Warm latency is about
47% higher than v4, but only 0.47 ms more in absolute terms. The matched control
still executes the head for cost comparison even though its selected correction
is zero; a deployed identity head would simply be removed.

Timing is on the local M5 Max, after training, using three fresh desktop WebKit
runs and both complete historical packed tracks per model. Warm block medians
exclude the first block; normal emission is 500 ms. They include the JavaScript
frontend and GPU readback, but exclude TTS, rendering, loading and audio scheduling.
Startup uses three fresh browsers per model with cached local assets/previously
compiled shaders, excludes import/rig/TTS and first-ever CDN network time.
Neither timing is an iPhone benchmark.

Full-track parity covered eight fresh engine/gain cases, two long packed tracks,
and a reset repeat of the first track. Across three runs, native/WebGPU maximum
coefficient delta was 7.01e-7 with no nonfinite, bad or unexpectedly zeroed frames.
The reset-repeat GPU tracks were byte-identical to their first generation for
all models. Five unit tests cover feature-time causality, cross-block history,
short tails, empty blocks, reset/initial zero correction and memoryless control.

The first browser packing attempt failed because ONNX Linear matrices had been
transposed/renamed as generated MatMul initializers. The corrected prototype
resolves their names from graph nodes and transposes their data for WGSL, with
source/model locks. Failed setup artifacts are retained; model weights, test
labels and selected checkpoints were not changed to repair this adapter.

## Sampled renderer/GPU footprint

Fresh desktop WebKit per model, normal James page animating idle; prerecorded
short speech, 27.5-second speech, then three repeated long utterances. Sample
`proc_pid_rusage` physical footprint every 20 ms and sum renderer/GPU at each
timestamp (not independent per-process maxima). No TTS session was prepared.

| Sampled physical footprint, MiB | v4 | Matched control | Temporal |
|---|---:|---:|---:|
| Short-generation interval peak | 495.0 | 486.2 | 490.7 |
| Long/repeated-generation peak | 528.9 | 525.7 | 523.4 |
| Last settled sample | 379.1 | 382.7 | 381.3 |

This single run shows no measurable page-level increase beyond run variability;
it does **not** establish that the temporal model saves memory. Explicit GPU
buffers do increase as shown above. Sampling can miss sub-20-ms spikes and
excludes the browser/network process. These are desktop inference-plus-rig
measurements, not full TTS+STA or physical-iPhone peak-memory guarantees. All
models settled after the repeated burst; no new unbounded growth was observed
in this bounded test.

## Visual review and decision

Sixteen cases were fixed before fresh scoring: /m b p/ for four presets plus
half-level /p/ and quarter-level vowel speech for both engines. Browser checks
passed synchronized audio, four James rigs, exact seek frames and no page errors.
Inspected screenshots and gap plots show similar student motion, without an
obvious broad closure improvement; this is not a blinded perceptual study.

Local review: `http://127.0.0.1:5182/.local/compact-temporal-v1/review.html`.

This is one seed and a frozen-backbone, approximately 117-ms feature-history
residual head. It does not rule out jointly training an encoder/decoder, longer
history, or a small recurrent motion state. It does show that this bounded
addition has no useful quality/cost justification. Keep v4 rather than deploy a
slightly smoother model that misses more closure events. Do not retune against
this now-inspected fresh set.

## Artifacts

[Protocol](compact-temporal-decoder.md) ·
[Reproduction](../../scripts/research/temporal_decoder/README.md)

Ignored root: `.local/compact-temporal-v1`, including frozen sources/protocol,
feature/data/normalization locks, optimizer checkpoints, selection/audit, full
native evaluations, browser source locks and three parity runs, startup samples,
physical-footprint traces and visual-review assets. Research checkout is isolated
from the shared main checkout; prior artifact symlinks are read-only inputs.

- v4: `5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd`
- Control: `0643b05f6de78516e6770158cc583886f20d7f6e3f05e350e715063c1f6e6b59`
- Temporal: `d9887175d0d3d3dc6e4d0a55b0944576ceef7a4ebb08374fb6c08f152194f62d`

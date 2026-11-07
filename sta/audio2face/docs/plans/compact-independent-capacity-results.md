# Independent channel capacity: results

The independently widened 128-channel model did not beat the matched 96-channel
control on fresh speech. Keep the published **v4 coverage-r2 (2.25 MB)**. The
research-only 3.22 MB export is not a new public v4 revision.

## Controlled comparison

Both arms started from coverage-r2, with the same 64-bin log-mel frontend, eight
residual blocks, context, data, replay, normalization, losses, seed 37, learning
rate and 744-update maximum budget. The 128-channel arm added independent random
features, preserving the original function through zero initial feedback/output
weights. All parameters were trainable. Both selected epoch 18 (558 updates)
on identical development data before fresh confirmation scoring.

The source/data locks and sampler audit passed. Initial development metrics were
identical. Extra encoder weights changed by 12.45% of their initial norm and
extra output weights acquired norm 0.1278. Thus the extra features learned; this
was not the earlier duplicated-channel experiment. Parameters rose from 331,047
to 572,455 (72.9%); ONNX download rose from 2,251,827 to 3,217,512 bytes (42.9%).
The budget matched updates and examples, not arithmetic or wall time.

## Fresh confirmation

24 previously unused transcripts × four training-held-out voice presets produced
96 new recordings (48 per engine), with original/half/headroom-boost/quarter
levels giving 384 teacher-labelled cases. The presets have appeared in previous
studies; the transcripts and recordings are fresh. Quarter-level speech was
absent from training and development. Test recordings did not select checkpoints.
Lip-gap RMS below is in James rig millimetres, macro-averaged over voices.

| Condition | Published v4 | Matched 96 | Independent 128 |
|---|---:|---:|---:|
| original | 2.6024 | 2.5524 | 2.5582 |
| half | 2.8965 | 2.8184 | 2.8184 |
| boost | 2.5875 | 2.5529 | 2.5463 |
| quarter | 3.3623 | 3.2987 | 3.2922 |

Original-level RMS worsened **0.23%** versus the matched control: Phonon worsened
0.60%, Supertonic improved 0.10%. Half/boost mean improved only 0.12%. The wider
model failed the frozen 5% improvement screen and the engine/contact safeguards.
Both extra-training arms improved over the published starting point, but that
is not evidence that widening helped.

| Original-level measure | Published v4 | Matched 96 | Independent 128 |
|---|---:|---:|---:|
| Contact F1 | 0.80031 | 0.80099 | 0.80037 |
| Opening-only RMS (mm) | 2.9746 | 2.8926 | 2.9108 |

Closure times use actual teacher contact events; phoneme windows identify the
corresponding event with up to 100 ms lead and 35 ms trail. They do not shift
teacher geometry onto acoustic phoneme onset. Timing MAE is conditional on
matched overlap events, so report hit counts alongside timing. All arms had the
same 93 /m/, 99 /b/ and 79 /p/ uniquely associated teacher events.

| Phone | 96 / 128 overlap hits | 96 / 128 well-timed hits | Onset MAE ms, 96 / 128 | Release MAE ms, 96 / 128 |
|---|---:|---:|---:|---:|
| /m/ | 68 / 68 | 60 / 58 | 12.01 / 12.25 | 12.50 / 13.97 |
| /b/ | 86 / 86 | 69 / 70 | 20.54 / 19.77 | 14.92 / 14.34 |
| /p/ | 59 / 59 | 50 / 52 | 11.02 / 10.73 | 13.56 / 12.99 |

## Previously inspected regressions

These sets are regression evidence, not fresh confirmation. Original-level RMS:

| Cohort | Published v4 | Matched 96 | Independent 128 |
|---|---:|---:|---:|
| historical | 2.7546 | 2.7278 | 2.7387 |
| coverage_regression | 2.6031 | 2.5700 | 2.5703 |
| Easton, 48 clips | 2.486 | 2.440 | 2.438 |

Historical contact and RMS guards passed. Small Easton gains did not establish
fresh cross-engine improvement.

## Browser cost and correctness

A research-only width-128 runner retained the existing two-kernel WebGPU design
and stable GELU. Production shaders, frontend and idle animation were unchanged.
Three fresh desktop WebKit runs each compared both models against native ONNX
on eight fresh gain/engine cases and both complete “packed” speech tracks.
Maximum coefficient delta was 7.90e-7; no nonfinite values, bad frames or
unexpected all-zero frames occurred.

| Cost | Matched 96 | Independent 128 |
|---|---:|---:|
| Explicit model GPU buffers | 25 | 25 |
| Explicit model GPU bytes | 1,492,236 | 2,496,652 |
| Warm block median, long speech | 1.00 ms | 1.15 ms |
| Range of six per-track medians | 0.96–1.08 ms | 1.12–1.20 ms |

Timing is on the local M5 Max in WebKit, after training ended, excluding the
first block per track. Each block normally emits 500 ms. It excludes TTS,
rendering, audio scheduling and startup. This is model-only speed, not end-to-end
latency or an iPhone result. Buffer counts exclude driver overhead, transient
allocations, JavaScript, renderer and TTS: **not a total-page peak measurement**.

## Artifacts and interpretation

Ignored artifact root: `.local/compact-capacity-v1`. Preserve `protocol.json`,
`selection.json`, `training-audit.json`, `evaluation-*.json`, `summary.json`,
`historical-evaluation`, `browser/source-lock.json`, and `browser/parity/run-*.json`
alongside full optimizer checkpoints, frozen sources, audio and teacher labels.

- Published coverage-r2: `5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd`
- Matched 96: `96535c26d4f3372584cceeb46f023e541b0638107c51c45b99dab71fbba9034b`
- Independent 128: `de706dc3e60360718acfe2706b33c4c9eb2f351f89c08d6ed4dc8f50183a14cd`

This is one seed and one fine-tuning budget. It does not prove a larger model
can never help, or rule out a different representation or training schedule.
It does show no practical capacity win for this controlled intervention. A
next isolated experiment should target articulation-state supervision or a
short temporal motion prior, retaining the efficient runtime frontend, rather
than widening further without evidence.

Reproduction: [research instructions](../../scripts/research/independent_capacity/README.md).
Frozen protocol: [experiment plan](compact-independent-capacity.md).

# Targeted difficult speech: results

**The context-targeted addition did not improve lip-gap RMS over equally sized
broad speech.** It had 0.0386% worse original-level RMS (0.00106 mm), slightly
lower contact F1 and two more well-timed closure hits. The predeclared screen
fails. Keep the newly released **v4 finetune-r4**; no model from this experiment
was published or deployed.

The requested ordinary-control release was completed first: CDN revision
`arkit-student-v4-finetune-r4`, source commit `78d8e31`, deployment
`dpl_53xPsaEsLQBNCZvzFbJDL6jx2T9D`, live at
https://james-audio2face.vercel.app/. Staged WebKit checks passed for Phonon,
Supertonic and Pocket, exact model hash/size, v4/WebGPU/Phonon defaults and idle
animation, followed by public-alias verification after promotion propagated.
Local v4 assets also match the release. Physical-iPhone stability is not claimed.

Read the [frozen plan](compact-targeted-speech.md) and
[reproduction instructions](../../scripts/research/targeted_speech/README.md).
The protocol/research code was committed as `33e603a` before fresh scoring.

## Target selection and controls

Start all arms from current v4 SHA
`ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da`.
The compact architecture/frontend/normalization/animation objective are unchanged;
there is no teacher-feature head or browser phoneme/voice input.

Mine current-v4 residuals on the existing **training windows only**. Aggregate
squared raw-teacher lip-gap error within aligned centre-phone intervals into
cross-word phoneme triples. Require eight distinct texts, 100 frames and presence
in at least three available candidate sentences. Rank RMS with the frozen
support-shrinkage rule; choose four bilabial, eight vowel and four other contexts.
All selected contexts have support across the eight training presets. Examples
include `d|b|ih`, `ah|m|ae`, `hh|ih|m` and `n|aa|t`. This is a text-level proxy for
acoustic difficulty, not actual-error mining of newly generated waveforms.

The retained inventory excludes 2,477 prior normalized texts, including all
feature/scale training/dev/test. There were 455 unused supported CMU ARCTIC
sentences. Reserve 80 by fixed hash for evaluation; use the remainder for training
candidates. Hash-select 72 broad texts first. Match 72 different targeted texts
to their word counts within one when possible, maximizing context score.
Training/development/test separation and source fingerprints passed the audit.

| Addition | Independent recordings | Mean selected-context score/phone triple | Mean words |
|---|---:|---:|---:|
| Broad | 288 | 0.07583 | 8.9306 |
| Targeted | 288 | 0.18670 | 8.7361 |

The targeted text score is **2.46×** broad. Scores are selection proxies, not
measured errors on these new utterances. Both arms use 72 texts × four voices,
paired voice/gain assignments, eight training presets and exactly 12 recordings
per voice/gain (36 per voice). Each new recording receives one independently
teacher-labelled original/half/headroom-boost level. A pre-synthesis draft caught
an imbalanced voice-cycle assignment; the final balanced schedule was checked
before any audio was generated, with the discarded draft retained locally.

Fresh confirmation uses 32 distinct reserved texts: 16 context-enriched with
bilabial/vowel examples and 16 hash-selected broad sentences, each in four known
training-held-out presets = **128 new recordings / 512 gain-labelled cases**.
Voices are previously studied, not new speakers. Original/half/boost/quarter
waveforms receive separate full-teacher outputs. Combined new corpus: 704
independent recordings / 1,088 teacher-labelled cases. Every shard's instrumented
teacher matched the untouched graph and assembly verified audio hashes, complete
tracks and identical recipes. New text is unchanged CMU ARCTIC; pinned source,
license and notice are retained locally. No original human speech is reused.

## Matched training

Three arms share seed 37, 3,720 AdamW/cosine updates, unchanged animation loss,
prior expanded speech/replay, frontend, parameter count and development selection.
Original-data control draws 48 existing primary windows plus 16 clean replay.
Broad/targeted keep the same 32 existing primary windows and replace 16 with new
voice-balanced examples. New examples are deliberately oversampled, not uniformly
mixed into the combined corpus. Broad has 2,115 novel cached windows; targeted
has 2,073. They differ in recordings/acoustics, not update count or new-data share.

Separate old/replay/new random streams preserve the old draws and replay despite
novel pool length differences. Complete old-draw and replay hashes match across
all three trajectories. Initial development results match exactly. MLX float32
with TF32 disabled and the previously validated export/backend are unchanged.
Checkpoint selection uses only common development plus Easton guard, initially
and every 124 steps. Selected steps: control 2604, broad 3720, targeted 3472.
All three exports were locked before fresh scores. Visual cases and evaluator
were fixed before scoring. No test-driven coefficient, seed, prompt or checkpoint
change followed the results.

## Lip sync on the new shared cohort

| Model | Original RMS mm ↓ | Contact F1 ↑ | Opening RMS mm ↓ | Well-timed /m b p/ total ↑ |
|---|---:|---:|---:|---:|
| Released v4 finetune-r4 | 2.77232 | 0.77097 | 3.03502 | 99 |
| Original-data fine-tuning control | 2.73106 | 0.76532 | 2.98094 | 104 |
| Broad addition | 2.73757 | 0.76908 | 2.99238 | 105 |
| Targeted addition | 2.73863 | 0.76788 | 2.99470 | 107 |

These are macro-voice scores on **this fresh cohort**. The same released v4 has
2.42607 mm on the earlier feature-study cohort; those numbers are not directly
comparable across distributions. Original-data fine-tuning improves RMS 1.49%
versus release; targeted improves 1.22%, but is 0.277% worse than that control.
Targeted versus broad is effectively tied in RMS, not a meaningful gain.

| Original-level stratum | Released v4 RMS mm | Control | Broad | Targeted |
|---|---:|---:|---:|---:|
| Broad reserved texts | 2.84789 | 2.81165 | 2.82438 | 2.82743 |
| Context-enriched reserved texts | 2.69479 | 2.64822 | 2.64882 | 2.64792 |

The enriched set was **not harder for released v4** than the broad set. Text
context enrichment is not calibrated waveform difficulty. Tiny improvement on
the enriched stratum does not compensate for the broad-stratum regression.
Targeted versus broad improves Phonon RMS 0.598% but worsens Supertonic 0.613%.

| Gain | Released v4 RMS mm | Control | Broad | Targeted |
|---|---:|---:|---:|---:|
| Original | 2.77232 | 2.73106 | 2.73757 | 2.73863 |
| Half | 2.91559 | 2.89447 | 2.88207 | 2.88502 |
| Headroom boost | 2.79188 | 2.74942 | 2.77420 | 2.74858 |
| Quarter | 3.25943 | 3.25717 | 3.24618 | 3.24576 |

Well-timed /m/, /b/, /p/ hits: release 37/27/35; control 40/29/35; broad 42/28/35;
targeted 41/28/38. This modest event-count gain coexists with lower framewise
contact F1. Retain the original anticipatory association and one-to-one event
rules; no phoneme-onset threshold was changed. Full association ambiguities,
conditional onset/release timing, engine/voice/source/category and velocity
metrics are in the reports. Timing errors must be read with missed-event counts.

Historical original RMS/F1: release 2.54635/.77603, control 2.44233/.78528,
broad 2.42125/.78419, targeted 2.47238/.77905. Inspected feature-regression RMS/F1:
release 2.42607/.80250, control 2.37224/.80184, broad 2.39168/.79956,
targeted 2.36346/.80163. Those are regression checks, not new confirmation.

The screen fails primary ≥3% improvement, engine non-regression, contact F1 and
broad-stratum RMS. Gain-average, closure-count and historical/feature guards
pass. Preserve `promising=false`. This fixed seed/selection proxy/budget does not
rule out every hard-example strategy; it does not justify a targeted-model release.

## Browser performance and bounded memory

All exports have **331,047 parameters / 2,251,827 bytes**, identical ONNX operators,
**25 model GPU buffers / 1,492,236 explicit bytes** and two reusable pipelines.
No runtime phoneme timing, new input or auxiliary head is added.

Twelve complete native-ORT/WebGPU fixtures × four models × three fresh runs,
plus first-clip repeats, give 156 comparisons. Max coefficient delta was 1.08e-6;
no nonfinite/unexpected zeroed frames were found and repeated first-clip GPU
outputs were byte-identical for every model. Sixteen synchronized five-lane
James cases passed audio and exact seeks, comparing fitted teacher, released
v4 and all three trained arms. This is functional review, not a blinded study.

A first browser attempt stopped at missing local rig files in the isolated
checkout (404). Restoring those asset links fixed the boundary; that failed
attempt is retained separately and contributes no timing or quality results.
The public deployment was unaffected.

| Model | Warm packed-block median ms | Observed packed medians range ms | Physical-footprint peak MiB, run1 / run2 |
|---|---:|---:|---:|
| Released v4 | 1.05 | 1.00–1.06 | 521.8 / 521.7 |
| Control | 1.02 | 0.98–1.04 | 525.2 / 520.7 |
| Broad | 1.09 | 1.02–1.12 | 524.6 / 526.2 |
| Targeted | 1.05 | 0.98–1.34 | 544.5 / 524.2 |

Timing is median of six packed-track medians per model across three settled
WebKit runs, after rig readiness and five warm-up calls. Each full block emits
500 ms of audio; this is desktop M5 Max STA cost, not TTS or iPhone latency.
There is no meaningful speed gain demonstrated for the unchanged graph.

Physical peaks sum new desktop WebKit WebContent+GPU processes concurrently at
20 ms during short/long/repeated prerecorded STA with normal James idle animation
active. **TTS is excluded.** An initial targeted peak ~23 MiB above v4 prompted
one bounded repeat; targeted measured 524.2 MiB on that repeat. Both observations
are retained. The excess did not repeat, and explicit GPU allocations stayed
identical; no repeatable weight-dependent memory increase or saving is established.
Browser/network processes are excluded and sub-20 ms peaks may be missed. These
measurements do not establish full-pipeline or physical-iPhone under-2GB safety.

## Artifacts and next hypothesis

Ignored root `.local/compact-targeted-v1` retains prior/text/corpus audits,
mined training errors/context scores, pinned source/license, audio/labels/maps,
source/training/selection locks, model/optimizer/RNG checkpoints, full scores,
three browser runs, both memory traces and synchronized `review.html`.
The local review used http://127.0.0.1:5183/.local/compact-targeted-v1/review.html;
use Vite's next printed free port on restart.

Training protocol SHA: `bfa532498bf87735dfd962ca262d62be9a1459eef85a40cac5756a539732b789`.
Control: `a1cf666862a59cb833af7df0b20a4af0b71fe29bded078623941c3feff646910`.
Broad: `b30d6364f21c6e69a7741d5d7b5f3bee1ca48813a242570ccfa561ead37553fb`.
Targeted: `a33a9942b24508a0b8b3d74488186ff8ccbadf3f9e1a0e1696834642ca2e09e5`.
All frozen source/input/selected-export checksums and sampling audits passed.

The next data hypothesis would rank **actual candidate audio** by validated
student/teacher residuals, with transcript/voice quotas and outlier checks, rather
than infer acoustic difficulty from phoneme triples alone. That would need a
separate frozen selection/control experiment and new confirmation recordings;
it was not tuned or run against this now-inspected set. Keep finetune-r4 live.

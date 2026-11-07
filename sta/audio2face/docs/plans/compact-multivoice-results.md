# Multi-voice pilot results — 2026-11-07

Study definition: [compact-multivoice.md](compact-multivoice.md).
Reproduction: [research README](../../scripts/research/multivoice/README.md).
These are research candidates, not replacements for the released demo models.

Subsequent three-seed and fresh-voice confirmation is recorded in
[the v4 validation results](compact-v4-validation-results.md). The exploratory
numbers and outstanding checks below describe the earlier pilot stage.

## What ran

336 fresh English recordings, 18.65 minutes, 16 preset voices balanced across
Phonon/Supertonic 3 and catalogue male/female labels. Eight training voices,
four validation voices and four test voices; all 52 prompts are partitioned by
text too. Training contains all 39 English phoneme types. Every recording has
fresh full-teacher geometry labels, a consistent lip-aware James fit, nine raw
lip gaps, optimized TinyAligner phone timings and teacher speech features.

Three matched arms used seed 37, 24 epochs, 31 optimizer steps per epoch,
64-window batches, 1,944 new training windows, the same initial v3 weights,
architecture, output normalization and learning-rate schedule. Selected epoch
was 18 in every arm. Auxiliary phone and teacher-feature heads are removed at
export. All exports have 331,047 learned parameters and are 2,251,827 bytes
(released v3 is 2,251,828 bytes; the one-byte export difference is not a compression
result). The browser model interface, kernels and 1,492,236-byte model GPU buffers
are unchanged.

The teacher's last audio-encoder layer was exposed in an offline backend copy,
without changing the original checkout. Untouched versus instrumented teacher
geometry matched exactly over 3,997,395 values. No FaceFormer download, paid
encoder API or additional browser speech encoder was needed.

## Accuracy on the four unseen voices

Headline scores below are equal-weight averages across the four voices, not
frame-weighted pooled scores. RMS measures nine vertical lip separations against
raw full-teacher geometry. Source units are centimetres; this table converts to
millimetres. Contact F1 is a framewise teacher-agreement metric, not human-rated
lip-sync accuracy.

| Model | Lip-gap RMS (mm) ↓ | Contact F1 ↑ |
| --- | ---: | ---: |
| Released v3 | 3.140 | 0.7215 |
| Multi-voice data only | 2.795 | 0.7520 |
| + training-only phoneme supervision | 2.794 | 0.7514 |
| + training-only speech-feature distillation | 2.793 | 0.7516 |
| Follow-up: multi-voice + clean Easton replay | 2.791 | 0.7485 |

Multi-voice fine-tuning reduced RMS about 11%. The two auxiliary tasks did not
provide a convincing additional test gain at these settings. Phoneme supervision
won the predeclared validation selection among the initial three arms, but its
test result was essentially tied with control. No combination was trained or
claimed as a winner.

The crossed voice/text bootstrap, 2,000 draws with voices resampled within engine,
gave the data-only arm versus v3 an RMS-difference interval of
[-0.677, -0.081] mm, and F1 difference [0.0028, 0.0537]. Auxiliary-task intervals
versus the data-only control included zero. This remains a one-seed pilot with
only two unseen voices per engine, not broad population or seed-robust evidence.

Frame-weighted engine/gender slices for data-only versus v3 also improved:

| Slice | v3 RMS (mm) | Data-only RMS (mm) | v3 / data-only F1 |
| --- | ---: | ---: | ---: |
| Phonon | 3.313 | 2.969 | 0.7267 / 0.7497 |
| Supertonic 3 | 2.998 | 2.628 | 0.7236 / 0.7641 |
| Male presets | 3.008 | 2.681 | 0.7618 / 0.7985 |
| Female presets | 3.302 | 2.918 | 0.6873 / 0.7121 |

## Historical-voice regression and the replay follow-up

On the older 48-clip Easton confirmation corpus, data-only RMS worsened from
2.580 mm to 2.734 mm; phoneme supervision reached 2.760 mm. This prevented
promotion of the initial pilot as a general replacement.

Two subsequent diagnostics were explicitly post-pilot, not new confirmation:

- **Matched Easton-only training:** same texts/optimizer-step budget, drawing
  training windows only from the pilot's Easton recordings. No epoch beat the
  initial checkpoint on the unseen validation voices, so epoch zero was selected
  and test output matched v3. The frozen generated recipe is archived locally.
- **Clean replay:** 48 new windows (six per training voice) plus 16 clean older
  Easton windows per update, using only the lip-aware v3/v3-extra targets.
  Legacy v1/v2 fits were excluded. The existing clean Easton development set
  became a fifth equal-weight validation voice. Selected epoch 17.

Replay retained the roughly 11% unseen-voice RMS gain while bringing historical
Easton RMS back to 2.597 mm, about 0.7% above v3; Easton F1 rose from 0.7618 to
0.7696. It is the engineering follow-up to inspect, with fresh confirmation and
independent seeds still required. Its SHA-256 is
`4fedbd2a948c7500e14b19809219c9997342b571a7590b0bb695fb19e33e7035`.

## Closure timing correction

The original event counter associated teacher contacts using 45 ms pre-padding
and 35 ms post-padding, then greedily assigned events to phonemes. It could omit
an episode that ended more than 45 ms before an aligned `/b/` or `/p/`, and it
could force a choice between neighbouring labials. Its previously quoted counts
are retained in JSON under the legacy policy, not used as the primary audit.

The corrected audit uses **100 ms pre-padding**, keeps the original geometry
timestamps, and reports shared/multiple/clip-truncated associations separately.
Padding tests 45/100/150 ms are saved. The padding applies to event association,
not to a permitted prediction error or a mandatory 50 ms animation shift.
An episode may begin earlier than the window if it continues into it.

Only uniquely associated teacher episodes lasting at least two 60fps frames
enter the primary phoneme-event denominator. A predicted closure may claim at
most one teacher episode. “Overlap hit” means any temporal contact overlap.
“Well timed” is a separate operational score: overlap IoU at least 0.5 and onset
error at most two frames (33.3 ms), relative to the teacher's actual closure.
Signed onset/release errors and exclusion counts are included in each report.
The time sampling remains 16.7 ms; the threshold is not a biological assertion.

| Sound | Unique teacher episodes | v3 overlap hits | Replay overlap hits | v3 / replay well-timed hits |
| --- | ---: | ---: | ---: | ---: |
| `/b/` | 51 | 28 | 42 | 24 / 34 |
| `/p/` | 62 | 29 | 49 | 24 / 39 |
| `/m/` | 43 | 22 | 28 | 19 / 25 |

These revised counts supersede the earlier 51/67/60-style denominators; they are
not direct before/after comparisons with the old greedy counter. RMS and
framewise contact F1 do not depend on the aligner and did not change.

Teacher closure onset medians in the uniquely associated test subset were about
93 ms before the TinyAligner `/b/` start and 100 ms before `/p/`; the tails extend
earlier. This measures geometry relative to CTC boundaries and includes possible
alignment bias. It does not establish human biological timing. Training mouth
targets always use unshifted teacher geometry, never a phoneme-centred closure
mask. The auxiliary phone task describes acoustics, not the prescribed time of
mouth contact.

## Browser cost and memory

Desktop M5 Max, macOS WebKit 2388 with iPhone 13 emulation. Phonon runs had two
fresh-browser repeats for each 4.2 s / 12.7 s clip. Supertonic had one repeat for
each 5.1 s / 15.1 s clip. All accepted runs had one navigation, audio playback,
ongoing rendered poses, and continued idle animation after speech. No training
ran during these measurements.

| Model / TTS | Runs | Median warm block (ms) | Maximum sampled peak (MB) |
| --- | ---: | ---: | ---: |
| v3 / Phonon | 4 | 3.00 | 909.1 |
| Phoneme pilot / Phonon | 4 | 2.93 | 902.0 |
| Replay / Phonon | 4 | 3.23 | 894.6 |
| Replay / Supertonic 3 | 2 | 2.81 | 1635.8 |

One block emits 30 usable frames, or 0.5 s of animation. These are animation-only
warm times including mel preparation/readback, not TTS generation or first-audio
latency. Timing/footprint differences between these identical-shape models are
not established speedups or memory reductions. The unchanged runner uses two
compute pipelines and 25 buffers. Replay CPU/WebGPU parity had maximum absolute
coefficient difference 2.69e-7 across 9,360 values.

Peak is the summed sampled physical footprint of WebContent and GPU processes at
20 ms intervals, not unique total machine memory. Supertonic was served from the
same pinned local assets because Vite does not implement Vercel redirects. Its
initial harness attempt failed visibly on missing assets and is excluded.
Desktop figures are not physical-iPhone guarantees or repeated-playback soak
results. Supertonic's full-page footprint is materially higher than Phonon's.

Six predetermined synchronized James comparisons are available from the isolated
research harness at `/review.html`. They show the fitted teacher, released v3,
three pilot arms and replay. They preserve one audio/frame clock and permit a
stationary-head scientific view; the normal demo and memory tests retain idle
animation. No blinded perceptual preference has been established.

## Further first-principles finding: gain sensitivity

A diagnostic on 16 **training** recordings (two categories from each training
voice) compared 0.5× / 2× waveform gain with the corresponding newly generated
teacher outputs, without clipping. Teacher lip-gap RMS changes were 0.216 /
0.161 mm; the multi-voice student's predictions changed 1.196 / 1.060 mm.
The compact frontend is substantially more gain-sensitive than the teacher.

However, the worst teacher clip changed 0.414 mm, exceeding the predeclared
0.05 mm invariant-label tolerance. Consequently **we did not reuse unchanged
teacher labels for gain augmentation**. This finding motivates a separate
properly labeled gain-variation corpus or carefully tested frontend normalization,
with a matching control. It is not a measured gain-normalization improvement.

## What to do next

Expand lexical/acoustic variety across the two engines while retaining clean
replay and unseen-voice validation. Test teacher-labeled gain variation as its
own arm. Repeat promising recipes across seeds and a new frozen confirmation
set, inspect opening amplitude as well as contact, and test on the physical
iPhone before selecting a release. The current evidence supports the data
distribution change; it does not justify claiming a breakthrough from either
auxiliary loss, replacing v3, or shipping a larger runtime encoder.

Raw manifests, asset/source locks, frozen first-pilot and replay recipes, labels,
feature caches, checkpoints, per-clip metrics and browser traces remain under
the ignored `.local/compact-multivoice` directory. Nothing was uploaded or deployed.

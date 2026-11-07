# Distinct speech coverage — first-seed results

Protocol: [compact-coverage.md](compact-coverage.md).
Reproduction: [research README](../../scripts/research/coverage/README.md).

## Outcome

The broader primary corpus improved new original-level lip-gap RMS **1.2%**
versus its matched old-corpus control, below the predeclared 5% screen.
Phonon improved 3.9%, but Supertonic regressed 1.2% against control and overall
contact F1 decreased slightly. The new model did not pass the engineering screen.
Published v4 gain-r1 remains unchanged; the coverage export is a research artifact.

This is not evidence that more data cannot help. It is an equal-clip,
equal-maximum-update-budget experiment replacing the primary transcript
distribution, with fewer voices per individual sentence and the same runtime
representation. It does not test an unlimited data/compute scale-up.

## Corpus and controls

The new primary training corpus has 128 original sentences, versus 32 in the
control. Each new sentence is spoken by one Phonon and one opposite-gender
Supertonic preset, rotating across the same eight training voices. Both arms
therefore have 256 base / 768 gain-conditioned primary recordings, 32 base clips
per voice, and identical clean historical replay. Original/half/headroom-boosted
levels are common; quarter level is held out from training/development.

The new corpus contains 416 freshly synthesized base recordings: 256 train,
64 development and 96 test, from 128/16/24 disjoint texts. Total base speech is
22.62 minutes: 13.57 training, 3.35 development and 5.70 test. Its gain variants
give 1,344 labelled recordings. Both arms use the same 288-recording development
cohort (old plus new) and clean Easton guard. Every changed waveform has fresh
full-teacher labels; no raw subtitle lines or dictionary-project animation
tracks were reused. Scalar-gain variants retain the waveform's time grid.

Teacher instrumentation again matched the untouched model exactly over
3,997,395 geometry values for each engine. All old/new labels share recipe
`112230e9acb49af828f9b7d626a99e2d2104304687feb362ca34eb067728872a`.
The completed audit confirmed identical initial development scores, equal
configuration/budget, exactly equal development tensor values, transcript split
isolation and eight-way training-voice balance.

Both arms initialize from the published v4 gain-r1 hash, use seed 37 and run the
same maximum 24 epochs × 31 updates, with the existing losses and selection rule.
The independently development-selected exports are epoch 15 for control and
epoch 17 for coverage; these hashes were frozen before test scoring. Both have
331,047 parameters and **2,251,827 bytes**. No runtime model/renderer changes occur.

## Text coverage is not model quality

The related OpenSubtitles analysis supplies only phone-pair frequency statistics.
The primary corpus increases distinct pairs 420 → 718 and triples 791 → 2,246.
Its subtitle-weighted pair occurrence coverage increases 74.31% → 93.03%.

However, clean historical replay already covers most common pairs. Including
replay, pair occurrence coverage is **99.69% → 99.76%**, distinct pairs
1,220 → 1,227 and triples 5,921 → 6,522. Total distinct primary-plus-replay texts
are 581 → 677. Reporting only the primary 74% → 93% figure would exaggerate
how much previously missing phonetic coverage this experiment adds. These are
text-derived counts, including cross-word transitions, not acoustic coverage
or evidence that every counted transition occurs without a pause in speech.

## New-transcript evaluation

The 24 new test texts were synthesized with four training-held-out pilot voice
presets and four levels: 96 fresh base / 384 variant recordings. The presets
were previously studied; only the transcripts and recordings are new. Neither
test results nor the previous v4 fresh-confirmation cohort selected checkpoints.

Values below are macro-voice RMS against nine raw-teacher lip gaps, in mm.
Positive percentages mean lower error versus matched control.

| New original speech | Live v4 gain-r1 | Matched control | Broader corpus | Improvement vs control |
| --- | ---: | ---: | ---: | ---: |
| All voices | 2.6703 | 2.6350 | 2.6031 | 1.2% |
| Phonon | 2.4832 | 2.4639 | 2.3668 | 3.9% |
| Supertonic | 2.8574 | 2.8061 | 2.8393 | -1.2% |

The broader model is 2.5% lower RMS than the live v4 on this fresh original-level
cohort, but part of that is also obtained by extra fine-tuning in the control.
Original contact F1 is 0.79005 / 0.79330 / 0.79089 for live/control/coverage:
a small gain versus live v4, a small regression versus control.

| New speech level | Control RMS | Coverage RMS | Control / coverage contact F1 |
| --- | ---: | ---: | ---: |
| Half | 2.8453 | 2.8514 | 0.77911 / 0.77610 |
| Headroom-capped boost | 2.6640 | 2.6267 | 0.79095 / 0.78870 |
| Quarter, unseen in training/development | 3.3772 | 3.3572 | 0.74987 / 0.74249 |

Half/boost mean RMS improves just 0.6%. Original opening RMS improves
3.0913 → 3.0262 mm and quarter-level opening RMS 4.0273 → 3.8954 mm versus
control, but contact F1 decreases. More accurate opening amplitude does not
establish better closure timing.

Well-timed original `/m/` hits are unchanged at 49/75; `/b/` rises 71 → 75 of
106 and `/p/` 47 → 50 of 89. Quarter-level hits worsen: `/m/` 42 → 40 of 76,
`/b/` 54 → 45 of 108, `/p/` 32 → 31 of 89. These use the unchanged unique
teacher-event association and actual closure onset/IoU criteria, not shifted
phoneme-centred targets.

## Historical checks and interpretation

On the separate prior gain-study original-level cohort, coverage RMS is
2.7546 mm versus live v4's 2.7359 mm: a 0.7% regression, within the 2% guard.
F1 falls 0.76072 → 0.75753, within the 0.01 guard. Historical 48-clip Easton
RMS improves 2.584 → 2.486 mm (about 3.8%) versus live v4, with F1 rising
0.7754 → 0.7822. The Easton gain does not imply both engines improved equally.

The screen failed fresh RMS, engine no-regression and contact no-regression;
it passed mean gain-level RMS and historical guards. Do not silently lower the
screen or retune against these test sentences. There is no independent-seed
confirmation, new browser memory/speed certification or physical-iPhone claim.

Twelve voice/category/level review cases were fixed before scoring. The four-lane
James viewer shares one speech/frame clock; WebKit verified audio and exact
snapshot seeks for all twelve. Selected renders show continuing underestimation
of teacher vowel peaks and no broad leap in detail. This is not a blinded study.

The next planned diagnostic is genuinely independent representation capacity,
with the same efficient audio frontend, rather than another nearly identical
channel replication or loss-weight sweep against this test set. Data coverage
remains important, but this bounded expansion has not explained the fidelity gap.

Artifacts are ignored under `.local/compact-coverage-v2`, including protocol and
source locks, raw synthesis, teacher labels, feature caches, full checkpoints,
frozen export selection, per-clip evaluations and synchronized review assets.
Coverage export SHA-256:
`5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd`.

# Teacher history and noise compatibility diagnostic

Completed 8 November 2026. Diagnostic only: no training or published model changes.
The released v4 finetune-r4 remains unchanged (331,047 parameters; 2,251,827 bytes).

## Question and answer

Can a stateless student match a teacher whose lips depend on previous speech?
Yes, there is a measurable compatibility problem: naturally reached teacher states
produce different lips for identical current audio windows. The effect is mainly
recent history. Noise is much smaller and is not a plausible explanation for most
of the remaining student error. This supports a small jointly trained recurrent
experiment; it does not establish a large achievable improvement.

## Frozen protocol and controls

See [predeclared protocol](teacher-history-compatibility.md) and
[reproduction instructions](../../scripts/research/teacher_history/README.md).
Inputs, source hashes, baseline and analysis were frozen before teacher runs.
Four-second clips were chosen before collecting results to obtain two eligible
suffixes for each of eight voices.

- 16 distinct four-second suffix texts, two per voice: Phonon Alfie, Easton,
  Maeve, Zoey; Supertonic F1, F2, M1, M2. Existing training recordings were
  selected by fixed hashes, without selecting on errors.
- Four histories per suffix: cold start, four seconds of silence, and two
  different four-second speech prefixes in the same voice. Prefix texts exclude
  all suffix texts. The two prefixes are shared by the two suffixes of a voice.
- Speech prefixes fade over their final 40 ms, ending with 20 ms of zeros.
  The common suffix has the same 20 ms initial fade in every condition.
- Seeds 0–3 per history: 256 factorial tracks plus 48 repeat/reset controls,
  for 304 teacher tracks. The existing stateless student was checked on all
  64 history inputs.
- Native teacher priming, recurrent state, one-second windows, 500 ms hops and
  frame crops are preserved. Noise is generated once per clip and reused across
  windows, as in the original teacher. Current training labels use fixed seed 0;
  they do not randomly resample noise at each inference.
- After 250 ms of common speech, corresponding one-second waveform inputs are
  byte-identical in all four histories. Earlier frames may still observe prefix
  audio and are excluded from the hidden-history claim. The main interval is
  250–3,250 ms, without end padding.
- Projection retains the actual 2,267 mouth/perioral vertices and nine lip gaps,
  not just a compressed loss. Against 3,997,395 untouched teacher output values,
  the selected surface and gaps were byte-identical, with zero maximum delta.
- All 16 cases passed exact repeat, identical-window and reset-state controls.
  Resetting state on each teacher call makes cold/speech-prefix outputs identical
  after 250 ms. The student also produces identical weights on matching windows.

These controls distinguish recurrent history from prefix leakage, sampling phase,
startup priming and nondeterministic execution. No nonfinite outputs were found.

## How much does history change the lips?

Fixed seed 0; pooled squared-error RMS across the three pairs among warm silence,
speech A and speech B. Surface RMS is Euclidean 3D displacement per vertex; gap
RMS averages nine scalar gaps. Both are expressed in physical millimetres in the
James rig coordinate system. They measure teacher-to-teacher differences, not
new student quality scores.

| Time after common speech starts | Gap RMS difference | Dense mouth 3D RMS | Contact disagreement |
| --- | ---: | ---: | ---: |
| 250–750 ms | **1.040 mm** | **1.276 mm** | **4.58%** |
| 750–1,250 ms | 0.148 mm | 0.186 mm | 0.139% |
| 1,250–2,250 ms | 0.037 mm | 0.047 mm | 0.139% |
| 2,250–3,250 ms | 0.007 mm | 0.007 mm | 0% |
| Whole 250–3,250 ms interval | 0.429 mm | 0.527 mm | 0.833% |

Contact means median gap at most 1 mm; disagreement is a frame classification
change, not an event-level /b/ or /p/ closure hit rate. The 95th percentile of
case/pair gap RMS in the first bin was 2.138 mm. There are 48 paired comparisons,
not 48 independent utterances.

The first matching bin differed by engine:

| Engine | Gap RMS | Dense 3D RMS | Contact disagreement |
| --- | ---: | ---: | ---: |
| Phonon | 0.350 mm | 0.428 mm | 0.278% |
| Supertonic | 1.429 mm | 1.753 mm | 8.889% |

The effect occurs across all eight voices. This small constructed sample cannot
establish a general engine ranking. Speech A versus B alone gives 0.897 mm gap
RMS in the first matching bin, so warm silence is not the sole source of the
result. Cold versus warm silence gives only 0.050 mm gap / 0.036 mm surface RMS
in that bin, and 0.021 / 0.015 mm over the primary interval: startup priming alone
is much smaller than the speech-history effect.

## Noise sensitivity

Comparing seeds 1–3 against seed 0 at fixed cold history gives **0.0175 mm gap**
and **0.0256 mm surface RMS** over 250–3,250 ms. With warm silence the numbers
are 0.0180 and 0.0264 mm. History/noise interaction is only 0.0059 mm gap and
0.0085 mm surface RMS over that interval. Repeat seed-0 execution is exact.
Noise sensitivity is small compared with the current student's approximately
2.196 mm gap / 2.652 mm surface error against the cold seed-0 teacher on these
specific clips. These diagnostic numbers do not replace the held-out benchmark.

## What does the decay mean?

We change history once, then feed increasingly shared audio. Convergence therefore
shows that old history is forgotten; it does NOT mean recurrence would matter
only once at the start of a real utterance. At each later student block, audio
before its finite current window is again unavailable. Missing recent history
can matter repeatedly. This experiment does not measure how frequently that
happens in ordinary uninterrupted speech or how much error recurrence recovers.

A supplementary exact squared-error decomposition uses an equal mixture of the
three warm histories. In the first matching bin, their spread around the teacher
mean is 0.601 mm gap / 0.737 mm surface RMS. This accounts for 5.13% / 7.65% of
student MSE on that constructed mixture; over the primary interval, 1.21% / 1.28%.
Most error on this probe remains conditional-mean prediction error. These are
not production variance shares, attainable rig optima, or expected training gains.
The supplementary decomposition and plots were written after primary results;
inputs and primary analysis were not changed.

## Next experiment

Test a jointly trained 32-unit causal state with the current audio frontend,
without more lookahead, against a matched stateless/reset-state control. Train
on sequential windows with explicit resets and evaluate both uninterrupted
speech and boundary/closure behaviour. Check free-running inference, browser
speed, persistent buffers and repeated-play peak memory before publication.

The earlier frozen-backbone 117 ms feature-history decoder and longer-window
trials are relevant negative/weak results, but do not test this same recurrent
hypothesis. Neither this diagnostic nor those trials justify promising a 10% gain.
Avoid investing in a large temporal model before the small controlled test.

## Artifacts

Local ignored directory: `.local/teacher-history-v1/`. It contains cases and
source hashes, frozen protocol and source copies, parity checks, 304 teacher
tracks, student checks, completion manifest, primary summary, supplementary
MSE decomposition, engine breakdown and two plots. Model/audio/track arrays are
not committed. The full run completed in about 130 seconds on this machine;
that is experiment wall time, not a browser speed or memory measurement.

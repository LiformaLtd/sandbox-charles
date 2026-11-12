# Distinct speech coverage follow-up

Freeze before synthesis, training and test scoring. Keep published v4 gain-r1
unchanged while running the study. This tests data distribution, not capacity.

## Matched intervention

Initial model: v4 gain-r1, SHA-256
`84da9bfe22d221075d85e8a52622fc7d31b18d12b04905e0902a7ec64eb5ab2f`.
Seed 37, existing losses/architecture/normalization, 24 epochs, exactly 31
updates/epoch and the same 48 voice-balanced + 16 clean historical replay
windows per minibatch. No new runtime encoder, aligner, phoneme head or renderer.

Control fine-tunes on the old 32 shared prompts, eight training voices, 256 base
recordings and three labelled gain conditions (768 recordings). Coverage replaces
that primary corpus with 128 new distinct prompts, each spoken by one Phonon
and one Supertonic voice of opposite catalogue genders, rotating equally across
the same eight training presets. This also gives 256 base recordings and 768
gain-conditioned recordings, with 32 base clips per voice. Historical replay is
identical in both arms. This is equal clips/updates, not equal total speech
duration or passes over windows; the new utterances may be longer.

The new original sentences cover closures, labial contrasts, broad vowels,
fricatives/affricates, dialogue and transitions/pauses. Definitions are in
`scripts/research/coverage/prompts.json`. No raw subtitles are copied. Use the
related dictionary project's OpenSubtitles phoneme-pair frequencies to report
text-derived coverage, not to select against held-out model errors. Report both
primary-corpus coverage and its union with clean replay; text phone-pair coverage
is not proof of recorded coarticulation or perceptual quality.

Keep the same pinned synthesis engines, eight denoising steps, speed 1.05,
Phonon temperature 0.7, per-recording deterministic seeds and resampler. Use
original/0.5×/headroom-capped boost conditions and regenerate corresponding full
teacher geometry labels. Quarter gain is evaluation only. Pin the existing
fitter/teacher precision/noise/timestamp protocol. Do not modify another project's
current fitter or aligner; reuse the prior frozen sources/assets.

## Development and evaluation

Both arms use the same development cohort: 16 new prompts across the four old
development voices at all three gains, plus the prior gain-study development
recordings. Select with the existing macro-voice gap/contact score and clean
Easton guard. Freeze selected hashes before held-out scoring.

Primary evaluation is 24 new prompts × four training-held-out pilot voices,
at original/half/boost/quarter levels: 96 fresh base recordings, 384 variants.
These voice presets were previously studied but not used for training; this is
fresh transcript/recording confirmation, not a claim of newly discovered voices.
Exclude every normalized prompt from all earlier training/evaluation manifests,
including v4 confirmation. Do not use the new test data for checkpoint selection
or repeated recipe tuning. Retain the prior gain-study test corpus and historical
Easton as separate regression checks, not fresh confirmation.

Report per-condition/engine/voice raw-teacher lip-gap RMS, contact F1, opening
and velocity errors, coefficient error and corrected /m b p/ closure timing.
The primary screen is at least 5% macro-voice RMS improvement on new original
speech versus the matched control, no engine-average RMS regression and no
contact-F1 regression. New half/boost average RMS must not regress. Historical
original-level RMS may regress at most 2% versus v4 gain-r1 and F1 at most 0.01.
Report quarter-level generalization separately. These are first-seed engineering
checks, not a release guarantee or population significance test.

If promising, repeat independently and confirm on additional untouched data,
review synchronized James rendering and verify WebGPU/native parity, speed and
repeated-playback memory without concurrent training. Physical iPhone validation
remains separate. If not promising, retain the result and move to an independent
representation-capacity experiment instead of tuning against the test set.

Artifacts: ignored `.local/compact-coverage-v2`. No model weights or raw audio
enter Git. Source/protocol locks must precede synthesis/training/scoring.

# Compact gain robustness: matched follow-up to v4

Frozen before training/evaluation. Existing v4 weights and the demo stay unchanged.

## Question and fixed contract

Can teacher-labelled loudness variation improve lip prediction without changing
the 96-channel architecture, frontend, model size, inference work or browser
memory? Prior training-only diagnostics found substantially greater gain
sensitivity in the student than the teacher. The teacher is not perfectly gain
invariant, so transformed recordings require their own full teacher labels.

Use the existing 256 training recordings from eight balanced Phonon/Supertonic
voices and consistent clean Easton replay. Conditions: original, 0.5× gain,
and amplification up to 2×, capped by the clip's available 0.98 full-scale peak
headroom. Never clip samples or amplify an already full-scale clip. Record
requested and actual gain and peak. Byte-identical transforms must not be counted
as new acoustic information. Regenerate original-level labels too, so all labels
carry the same current source fingerprint.
Timing and phoneme identity remain unchanged under scalar amplitude transforms.

## Matched arms

Both initialize from the frozen released-v4 representative (SHA-256
`b4a275acf78b982fb6ec6fc55333c5cf25613e72b6d9b3d80e95d6e74ac6261d`).
Use seed 37 initially, 24 epochs, exactly 31 updates/epoch, the existing losses,
learning-rate schedule, batch size 64 and clean replay (48 voice-balanced new
windows plus 16 clean historical windows). The only intervention is the training
recording distribution: original versus original/half/headroom-boosted. This is
an equal-update comparison, not equal passes through the differently sized corpora.
Different corpus sizes can produce different sampled windows at the same RNG seed.

Both arms select checkpoints on the **same** development recordings across all
three gains, with the same macro-voice gap/contact score and historical clean
Easton guard. The original v4 and the matched original-only fine-tune are both
baselines. Training labels retain the pinned teacher, precision, noise, James
identity, fitter and streaming timeline. Auxiliary heads remain unused/exported
away; no phoneme aligner or new speech encoder is added at runtime.

## Evaluation and gates

Freeze definitions and source/audio/model hashes before training. Use the pilot's
48 held-out recordings at original/half/headroom-boosted levels, plus a 0.25×
condition absent from training/development. These speakers/texts are disjoint
from training, but their original recordings were previously inspected in the
v4 study: report this as a gain follow-up, not fresh voice/text confirmation.
Do not tune against this evaluation or reuse v4's fresh confirmation set for
selection. Report per-gain/engine/voice lip-gap RMS, contact F1, coefficient error,
opening/velocity errors and the existing corrected /m b p/ closure audit.

An initially promising arm must reduce macro-voice RMS on the two trained gain
conditions by at least 5% versus the matched control, with no engine-average
regression. Original-level macro RMS must regress no more than 2% versus v4,
and original-level contact F1 by no more than 0.01. Quarter-level generalization
and historical Easton regression are reported separately. Gains limited to 1
because of no upward headroom must be reported separately from amplified clips.

Only if promising: repeat with independent fine-tuning seeds, generate genuinely
new voices/texts for confirmation, review synchronized James articulation, verify
native/WebGPU parity, block speed and repeated-playback memory without concurrent
training, then request physical iPhone review. An equal topology does not itself
establish a new memory measurement. Do not promote an experimental model based
on the first seed or gains on this previously inspected cohort.

Artifacts are ignored under `.local/compact-gain-v2`: protocol/source locks,
transformed audio, manifest, teacher labels, arm directories, checkpoints and
evaluation. Research scripts are under `scripts/research/gain/`. Existing study
artifacts and models remain untouched.

Setup v1 stopped before training because the cache guard detected a labeler
source-hash difference. Inspection found only progress-log changes, but the guard
was not bypassed. Setup v2 regenerates all labels under one current recipe instead
of mixing cache provenance. No model was trained or test-scored in setup v1.

# Compact v4 candidate validation — frozen 2026-11-07

Validate the multi-voice + clean-replay recipe without changing its architecture,
training data, loss weights, learning rate, epoch budget or checkpoint-selection
rule. This follows the exploratory pilot; its old test sets are regression tests,
not fresh confirmation. Existing released models remain unchanged.

## Training and representative selection

Repeat seed 37's exact recipe with seeds 73 and 101: released-v3 initialization,
96 channels, 24 epochs, 31 steps/epoch, 48 new multi-voice windows plus 16 clean
v3/v3-extra replay windows per update. Keep all input/source hashes. These are
independent fine-tuning seeds from the same pretrained initial model, not three
complete from-scratch training histories. Use the five-voice development cohort
and the existing gap/contact selection score for each checkpoint.

Before looking at fresh confirmation scores, select a representative by minimum
validation `best_selection` across the three completed runs; resolve ties by
seed number. Report every seed and the mean, not just the representative. Freeze
the selected model hash and validation scores before evaluating confirmation.

## New confirmation data

Use 24 newly written texts across six categories: `/m/`, `/b/`, `/p/`, broad
vowels, fricatives and conversation/pauses. Generate all texts with four voices
absent from the pilot and known compact training: Phonon Trevor (male, US),
Phonon Pippa (female, UK), Supertonic M3 and F3. That makes 96 recordings, with
equal engine/gender/text representation. No pronunciation guessing: check the
frozen dictionary and exact-normalized transcript non-overlap before synthesis.
Change an unsupported word before generating any data, not after seeing scores.
The full Audio2Face teacher's original training speakers are not known; the
holdout claim concerns the compact student's recorded training history.

Keep model/voice revisions, TTS settings, resampler, optimized TinyAligner,
James fitter, teacher seed/precision and speech-feature alignment identical to
the pilot. Source hashes for shared components must match. Timed phonemes are
offline-only. Use the corrected 100 ms association audit, ambiguous/shared
exclusions, separate overlap versus well-timed hits, and unchanged raw teacher
geometry timestamps.

Source definitions are `scripts/research/multivoice/confirmation-{voices,prompts}.json`.
Raw artifacts go to ignored `.local/compact-v4-confirmation`; model validation
and browser artifacts go to `.local/compact-v4-validation`. The protocol and
planned-corpus hashes are frozen before the first confirmation evaluation.

## Analysis and engineering screen

Primary: mean across seeds of macro-voice lip-gap RMS versus the fixed released
v3 model. Also report contact precision/recall/F1, corrected closure timing,
opening/transition errors and per-engine/gender/voice/category slices. Use a
paired crossed bootstrap over seed, voice and text, resampling voices within
engine. Three seeds and two voices per engine remain a limited uncertainty
estimate. Do not tune parameters against this confirmation set.

Predeclared engineering checks, applied to mean candidate scores:

- Fresh macro-voice RMS improves at least 5%, with no engine-average regression.
- Fresh macro contact F1 does not regress.
- No fresh voice's RMS regresses more than 2%.
- Older 48-clip Easton mean RMS regresses no more than 2%; F1 delta is at least
  -0.01. Keep those historical results separate from fresh confirmation.

These point-estimate checks are not significance tests or proof of universal
voice coverage. Report confidence intervals and failed checks explicitly. A
passed screen makes a candidate worth on-device review, not an automatic release.

## Runtime and visual validation

Use the validation-selected representative, the existing dedicated WebGPU runner
and unchanged renderer. Check numerical CPU/WebGPU agreement before timing.
Compare released v3 and the candidate with Phonon and Supertonic in separate
fresh WebKit browser contexts. In each context, prepare both models once, then
generate three **different** texts on the same page so the identical-text replay
shortcut cannot hide repeated-generation allocations. Verify audio, frame
progression, no navigation/reload, continued idle animation, and stable model
GPU allocation counts. Sample WebContent/GPU footprint throughout the sequence.
Do not run training concurrently with browser timing/memory measurements.

Use six predetermined fresh visual cases (frozen in the artifact protocol) for
synchronized teacher-fitted/v3/candidate rendering with shared audio and clock.
The scientific view may hold the head steady; the normal demo and runtime test
must keep idle motion. Visual inspection is not a blinded perceptual study.

Physical iPhone testing remains a separate user/device step. Desktop WebKit's
memory footprint, including iPhone emulation, does not certify an iPhone budget.
Do not deploy or replace v3 merely because desktop checks pass.

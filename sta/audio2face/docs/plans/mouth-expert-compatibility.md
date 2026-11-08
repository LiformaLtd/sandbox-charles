# Frozen mouth-expert compatibility diagnostic

Before any James expert scores: no expert, animation or student model training.
Primary expert: frozen pretrained audiovisual Auto-AVSR checkpoint distributed
by the official INTERSPEECH2024 AV-guidance repository. Pin its authors' Auto-AVSR
commit8acf580bcab10507532854ca5371f76ca1c364a5 and original state dict/hash. Load
all parameters strictly on CPU, eval mode, no masking/dropout/augmentation.
Primary score is its joint CTC(.1)+attention(.9) transcript loss; lower is better.

Use existing full-teacher-fitted James ARKit outputs for12 distinct transcripts
from the last confirmation corpus: four calibration and eight test recordings,
one calibration and two tests per known held-out preset (Phonon Freya/Hugh,
Supertonic F5/M5). Selection is salted text hash without expert outputs, requiring
adequate duration/openings and phoneme-associated reference contacts. These are
new cases for the expert, not previously unseen speech/model-training voices.
Resample coefficient trajectories60→25fps by linear interpolation; truncate audio
to matching640-sample frames. Freeze gaze/brows/head/blink and camera/lighting.
Use the same Three.js James rig and packed blendshapes as the demo.

Variants, fixed before scores:
- Correct fitted reference.
- Visual sequence shifts−320,−160,−80,+80,+160,+320ms, audio unchanged. AVSR uses
  clamped edge frames; SyncNet uses common interior windows without boundary pads.
- Under-open: reduce jawOpen by up to50%, targeting half the opening above1mm,
  preserving all reference closed frames; no lip inversion or abrupt threshold.
- Missing closure: only uniquely associated /m,b,p/ teacher contacts (same100ms
  lead/35ms trail association), fitted contact frames lifted above3mm using
  jawOpen +.18, with80ms shoulders. Silence is not the target corruption.
- Static midframe pose, secondary strong negative.
Geometric integrity/visibility are checked before scoring, not chosen by scores.

Render224×224 RGB mouth-view and face-view PNGs, DPR1, no compression. AVSR mouth
crop128×128 centered, resized88, grayscale then (x−.421)/.4 as paper code. Audio
is per-clip layer-normalized16kHz. Transcript cleaned to uppercase letters/spaces
and tokenized with the exact bundled SentencePiece units. No text/audio changes
between paired animation variants. Raw loss magnitude is not a recognition WER.

Predeclared secondary modes: checkpoint-default std.165 instead of.4; and paper
std.4 with muted audio to expose audiovisual domination. These are reported
separately, not selected after seeing results. Secondary SyncNetv2 uses its
original MFCC/BGR preprocessing and frozen weights. Render face crop[16:176,
32:192]→224. Choose one global offset in±200ms using only four correct calibration
clips. Report both calibrated and zero-offset fixed scores. Never minimize over
shift per corruption, which would make shifted animation look correct.

Verify strict model loads, finite outputs, direct-versus-cached loss parity,
repeated-score determinism and same-length reversed-token negative controls.
Freeze input weights/audio/track/source/model hashes before primary scoring.

Compatibility gate on eight test clips, separately for each expert/mode: correct
wins by>1e-5 loss/distance on at least6/8 clips for each±160/±320ms shift,
under-opening and missing-closure; each engine at least2/4 wins for each required
corruption. Report±80ms/static without using them to choose a scoring offset or
threshold. Report paired deltas, counts and every case, not just pooled means.
An overall pooled win cannot hide a failed corruption/engine. Primary is paper-AV;
a passing secondary is a hypothesis requiring independent confirmation.

This is a compatibility screen, not proof of useful gradients, training benefit,
human perceptual quality or browser efficiency. Expert/checkpoints stay offline.
Do not train against or publish any expert that fails the mouth corruption gate.

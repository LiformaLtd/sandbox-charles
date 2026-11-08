# Frozen mouth-expert compatibility on James

Completed 8 November 2026. **SyncNet is a promising training-only auxiliary expert;
the tested audiovisual transcript-loss expert is not reliable on James as-is.**
No expert/student training, model replacement, CDN upload or deployment occurred.

[Initial frozen protocol](mouth-expert-compatibility.md) ·
[Reproduction](../../scripts/research/mouth_expert/README.md)

## Experts and implementation

The primary expert is the publicly distributed pretrained Auto-AVSR checkpoint
linked by the [official AV-guidance paper implementation](https://github.com/kaist-ami/3d-talking-head-av-guidance#lip-reading-expert).
It was loaded strictly on CPU, in eval mode, all parameters frozen. Its score
is the original joint .1 CTC + .9 attention transcript loss; lower is better.
Correct transcripts use the bundled uppercase SentencePiece model/IDs. No loss
values are claimed to be word error rates or calibrated probabilities.

The paper-code grayscale normalization std.4 was primary. Two predeclared
secondary modes use checkpoint-default std.165, or std.4 with muted audio.
The paper's training code also updates the lip-reader and masks audio; this
study deliberately does neither training nor adaptation. It tests the public
frozen checkpoint for the proposed use, not the paper's final adapted system.

Secondary [SyncNetv2](https://github.com/joonson/syncnet_python) compares audio and
five-frame visual embeddings by Euclidean distance. We retained original16kHz
MFCC/int16-audio and BGR0–255 preprocessing. Correct/corrupt comparisons use the
same interior windows. We report zero-offset scores and a single global offset
chosen ONLY on four correct calibration clips. No per-variant best-offset search
is allowed: that could undo the deliberately introduced timing corruption.

The fixed global alignment is−1 visual frame (40ms at25fps). This aligns scorer
features; it is not a request to delay the production avatar or move phoneme
labels. The unmodified official real-video example recovered the expected3-frame
offset, minimum distance5.3518 (README approximately5.353), confidence10.087
(approximately10.021). Both expert state dicts loaded strictly; cached-feature
AVSR losses matched direct forwards exactly, and same-length reversed-token
transcripts had higher loss in12/12 controls. This is an implementation control,
not proof that AVSR pays attention to the mouth.

Pinned source commits:

- AV-guidance: `7e4c63bb4f9da8d3d932d19c89bc1c34d25cb70e`.
- Auto-AVSR: `8acf580bcab10507532854ca5371f76ca1c364a5`.
- SyncNet: `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623`.

Checkpoint SHA256:

- AVSR: `16304660a2181629215f8c7390565750784379f6e8a2b5c8d155fd9574afcd6e`.
- SyncNet: `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`.

## Cases, geometry and rendering

Initial set: four calibration and eight test clips from the previous loss-balance
confirmation corpus, one calibration/two tests per Phonon Freya/Hugh and
Supertonic F5/M5. All twelve transcripts differ. Salted-hash selection used
available duration/openings and verified /m,b,p/ contacts, never expert outputs.
These are new cases for this test, not new voices or guaranteed unseen examples
from the experts' original training data.

Reference is full-teacher output fitted to James blendshapes, linearly resampled
60→25fps. Audio is truncated to matched640-sample frames. The actual Three.js
James rig/materials/packed blendshapes render lossless224×224 PNGs. The mouth
view's centered128×128 crop is resized to88; SyncNet uses a fixed face crop resized
to224. ROI setup was visually inspected before any James expert scores. Head,
gaze, eyebrows and blinking are fixed so they cannot become shortcuts. Both
local/CDN blendshape hashes match the frozen geometry assets.

Corruptions preserve the same audio:

- Visual shifts ±80/160/320ms. AVSR uses clamped edge frames. SyncNet uses a
  common interior that never needs boundary padding, including its fixed offset.
- Reduced opening: jawOpen reduction capped at50%, targeting half the opening
  above1mm, while preserving all existing closed frames. The measured median
  reduction on open frames is about1.62–3.46mm on the first set.
- Missing closure: only uniquely associated /m,b,p/ contacts, using the existing
  100ms lead/35ms trail association, are lifted to gaps above3mm with80ms
  shoulders. Silence is not the target. Minimum targeted gaps exceed3.42mm.
- Static midframe pose, an additional strong negative.

Cases/weights/audio/sources/checkpoints were frozen before scores. Initial
engineering drafts were retained: NumPy/Accelerate matmul warnings were removed
using explicit einsum, and unrestricted contact lifting was replaced with
phoneme-associated contacts, both before scoring/freeze. No expert result guided
these fixes or the crop. Rendered QA confirms visible reductions and removed seals.

## Initial test: correct animation preferred, out of eight clips

| Corruption | Primary AVSR | Fixed-offset SyncNet |
| --- | ---: | ---: |
| Delayed320ms | 5/8 | **8/8** |
| Delayed160ms | 5/8 | **8/8** |
| Advanced160ms | 3/8 | **8/8** |
| Advanced320ms | 3/8 | **8/8** |
| Reduced opening | 4/8 | **8/8** |
| Missing /m,b,p/ closure | 4/8 | **8/8** |
| Delayed80ms, secondary | 5/8 | **8/8** |
| Advanced80ms, secondary | 3/8 | **8/8** |
| Static pose, secondary | 5/8 | **8/8** |

The gate requires at least6/8 correct preferences for EACH ±160/±320ms and
opening/closure corruption, plus at least2/4 for each engine. Primary AVSR fails.
Changing to std.165 does not rescue it: reduced opening2/8, missing closure3/8.
Muted-audio AVSR improves closure discrimination to7/8 but reduced opening is3/8.
A muted whole-transcript recognizer cannot itself establish audiovisual timing;
its asymmetric shift results must not be interpreted as a valid sync metric.
No AVSR mode passes the frozen complete gate.

Calibrated SyncNet passes every family/engine: Phonon4/4 and Supertonic4/4 for
each corruption. Average corrupted-minus-correct distance is+1.541 for reduced
opening and+0.609 for missing closure; positive means correct is preferred.
Without calibration it still passes the required gate, with7/8 opening/closure
wins, although delayed80ms wins only4/8. This supports fixing alignment once
rather than silently choosing a new offset for each example.

## Independent confirmation of the selected secondary expert

Because SyncNet was initially secondary, we did not retrospectively declare the
primary AVSR hypothesis successful. A separately frozen post-screen confirmation
selected eight distinct texts/recordings from the earlier feature-confirmation
corpus, two per same four voices, excluding all twelve initial transcripts.
Checkpoint, ROI, corruptions, thresholds and−1-frame global offset are unchanged;
no recalibration, tuning or training was performed.

**Correct animation wins8/8 for every corruption again**, including both engines
4/4, all six timing shifts, reduced opening, missing closure and static pose.
Average opening/closure distance penalties are+1.572/+0.761. Across the two test
sets, SyncNet correctly ranks16/16 clips for each corruption:144 paired wins,
but only16 independent clips, not144 independent utterances. The sets total
about49.5seconds of test speech. This is bounded evidence on four known TTS
presets and one rig, not a general human-perception validation.

The input/frame audit checks all20 calibration/test/confirmation cases, frozen
hashes, valid coefficient bounds, unchanged non-jaw channels, preserved reference
closures, PNG dimensions and nonzero contact-corruption coverage in every tested
SyncNet window set. One calibration clip's target contacts lie outside that
interior; it was used only as a correct calibration track, never for a corruption
win. No uninformative test case was discarded to improve counts.

## Decision and limitations

Use frozen SyncNet as the candidate for a **new matched training experiment**,
retaining teacher geometry/contact supervision and unchanged browser architecture.
Do not use the tested frozen audiovisual transcript loss as the primary mouth
quality guide. A low recognition loss can coexist with wrong motion; recognizing
text is a different objective from judging synchronized articulation. The results
are consistent with audio dominance but do not isolate its contribution exactly.

Compatibility is not yet a quality improvement. The experiment does not prove
useful gradient directions, submillimetre sensitivity, resistance to exploitation
by a trained generator, full lip readability or a gain over v4. Corruptions are
moderate and deliberately constructed. Before using SyncNet in optimization,
validate renderer/expert gradients and select its coefficient on training data;
then compare with matched control on new confirmation speech and the existing
lip-gap/closure/engine guards. Do not select or report a model gain on these
already inspected compatibility clips.

All experts/checkpoints remain offline. No neural weights were trained, no new
browser runtime was added, and v4/WebGPU/idle behaviour was not changed. Primary
scoring took about109seconds on CPU; this is not browser speed or memory evidence.
Large models, rendered PNGs, features, score JSONs and figures remain ignored in
`.local/mouth-expert-v1/`. Source, protocol and learnings are retained in Git.

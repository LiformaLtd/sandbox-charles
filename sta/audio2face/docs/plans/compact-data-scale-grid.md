# Frozen primary data × training-budget grid

This follows the passed four-clip memorization screen; those training-only fits
are not deployable or held-out quality evidence. Published v4 coverage-r2 stays
unchanged. Freeze this grid before optimization or fresh confirmation scoring.

## Data

Existing primary: 256 original recordings / 768 gain-conditioned examples.
Expansion: add 3072 newly synthesized recordings, using the same eight training
presets (Phonon/Supertonic, balanced engines and catalogue genders). This gives
3328 independent primary recordings (13×), and 3840 primary training examples
(5×). Each new recording receives one of original/half/headroom-boost levels,
exactly balanced per voice. Old recordings retain their three existing variants.
New levels receive their own full-teacher geometry; labels are never copied
between changed waveforms. The amplitude marginals match, but this tests a
corpus-expansion bundle, not volume independently of linguistic distribution.

New primary text: 512 unchanged CMU ARCTIC prompts + 256 authored articulation/
dialogue prompts, four training voices per text. The 549 clean historical replay
recordings stay unchanged; 13× does not describe the total including replay.
Keep the CMU notice, source commit/hash, selected texts and provenance outside
Git. No original human speech recordings are reused.

Common development: old 288 examples + 192 new recordings × 3 levels = 864
examples, four training-held-out presets, plus the unchanged clean Easton guard.
Fresh confirmation: 64 additional ARCTIC + 24 authored texts, four known held-out
presets = 352 new recordings / 1408 original/half/boost/quarter cases. Presets
are previously studied, not new speakers. Exclude all retained prior normalized
texts, not just training texts. Freeze visual cases before fresh scoring.

## Compute and controls

Initialize all from published v4 coverage-r2 SHA
5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd.
The architecture/frontend/precision/channel map/normalization remain unchanged.
Use MLX 0.31.2 native GPU training after recorded CPU forward/loss/gradient checks,
then export through the unchanged PyTorch model to the same 2,251,827-byte ONNX
interface. Source/backend hashes and numerical tolerances are locked. No Torch
MPS training is used: a preflight showed inconsistent gradients/loss trajectories.
This new grid compares only arms using the same validated backend.

Two trajectories (original and expanded), each 3720 updates, produce four arms:
original-short / original-long / expanded-short / expanded-long. Short budget is
744 updates; long is 3720 (5×). Short is an exact prefix of its long trajectory,
with a frozen selected checkpoint at step 744. Shared LR schedule: cosine .00012
to .000012 over all 3720 updates, AdamW beta .9/.999, epsilon 1e-8, bias correction,
weight decay .015, gradient norm clip 1.0. Thus the short schedule differs from
previous 744-step studies, but this grid's budget comparisons share identical
prefix updates instead of changing the LR schedule too.

Use seed 37, numpy PCG64 sampler seed 1037, 48 balanced primary windows (six per
voice) + 16 clean historical replay windows per update. Draw windows uniformly
within voice. Each update uses 30-frame valid crops from the unchanged 1-second
input. Full existing coefficient/geometry/contact/velocity/timing objective.
Keep the normalization statistics and all 331,047 trainable parameters unchanged.

Evaluate initially and every 124 updates, including step 744 and 3720. Select
minimum macro development voice gap RMS + .25*(1-contact F1), including Easton as
fifth voice. Short selects only through step 744; long through step 3720. An
initial/short checkpoint may win; never force a later checkpoint. Save optimizer,
RNG, source/data hashes and both selected states, locking exports before scoring.

## Acceptance and reporting

Primary comparison: expanded-long versus original-long. Require >=5% fresh
original-level macro-voice lip-gap RMS improvement, no engine-average RMS or
frame-contact F1 regression, no half/boost mean RMS regression. Total well-timed
/m b p/ hits must not fall; no category may lose more than two well-timed hits.
Report original/half/boost/quarter, opening error, onset/release timing, category
and engine results, comparing every arm with v4. Historical/coverage original
RMS may worsen at most 2% and F1 at most .01 versus v4. No test data chooses
checkpoints, budgets, source mix, thresholds or learning rate.

Also report data effects at both budgets, compute effects for both datasets and
any interaction. This is one seed; independently confirm a promising recipe
before release claims. If it fails, keep the negative result and do not tune
against this confirmation set. If the fixed pilot comparison passes, repeat both data trajectories with seeds
73 and 101 using the same caches and schedule. Do not alter the recipe or choose
a seed against confirmation. Report individual and mean data effects; every
repeat must improve original RMS and the mean must retain the >=5% improvement
with closure/engine guards. These repeats assess seed stability on the same
confirmation cohort, not a second independent test population. No deployment in
this phase.

Verify unchanged ONNX topology/size, native/WebGPU full-track parity, warm block
cost, bounded repeated-generation allocations and idle animation. Physical
renderer/GPU footprint tests use prerecorded audio with James idle and no TTS;
report them separately from full TTS+STA or physical-iPhone peak guarantees.
Benchmark only after training/teacher generation ends.

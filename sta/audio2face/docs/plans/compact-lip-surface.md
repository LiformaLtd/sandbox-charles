# Dense lip-surface supervision: frozen pilot

Keep published v4 finetune-r4 unchanged. Initialize both arms from SHA
 ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da.
No new browser architecture, inputs, encoder, state, precision or runtime graph.
Expected331047 parameters /2251827 bytes /1492236 explicit GPU bytes.

## Geometry and exact loss

Fix the existing James mouth-region bounds on neutral teacher skin:
abs(x)<3,166.1<y<169,z>5 cm, giving2267 vertices. This dense mouth/perioral region
includes all18 established lip-gap landmarks and inner/outer mouth skin, not a
claim of a manually segmented vermilion annotation. Supervise all three axes
against RAW full-teacher geometry, not merely fitted coefficients or nine gaps.
Rig prediction is R+wB and target is S+D; compare wB to t=D+(S−R). Never discard
the neutral offset or store absolute head-height coordinates in float16.

The rig is linear. Per axis retain q=tBᵀ/V and s=mean(t²); Gram G=BBᵀ/V is fixed.
Then per-axis MSE=wGw−2wq+s exactly equals dense vertex MSE. Full3D vertex RMS
is10sqrt(mean(sum_axis MSE)) in mm. Use float32 statistics and compare dense/
quadratic values and gradients on real frames. This is exact sufficient-statistic
compression, not dimensional approximation or a reduced landmark objective.
An offline backend copy projects only the HOST transfer, keeping the original
teacher graph/GRU state and window schedule. Verify it against untouched teacher
geometry before labelling. Check every regenerated track's nine gaps against its
existing pinned references, including changed gain waveforms and end crops.

## Training and controls

Reuse audited expanded primary corpus (3328 originals/3840 gain cases), unchanged
common development and549 historical replay. Regenerate raw surface statistics
for primary train/common-dev only; replay retains ordinary animation losses.
Both arms keep the same331047 parameters, initialization, all animation losses,
normalization, sampled batches, seed37 and3720-update AdamW/cosine schedule.
Draw48 primary windows, six per eight voices, plus16 historical replay as before.

Control surface coefficient0. Experimental adds mean summed-axis surface MSE
on valid PRIMARY frames only. Calibrate a single coefficient on one frozen,
voice-balanced training minibatch so its initial backbone gradient norm is25%
of the ordinary objective gradient norm; ramp to it over372 updates. Freeze
that coefficient before optimization, with no sweep or confirmation tuning.
No surface targets/replay changes other than this added primary loss.

Use validated MLX float32/TF32 disabled; compare new loss/gradients against CPU
and dense geometry. Development selection is identical in both arms: mean over
four common-dev voices of gap RMS+.5*full3D surface RMS+.25*(1−contact F1), with
the unchanged Easton guard contributing its gap/contact score as fifth voice.
All RMS terms are in cm inside selection. Raw surface is absent for Easton.
This joint selector is new but shared; it prevents selecting only nine-gap gains
when the purpose is visible surface shape. Evaluate initial/every124, allow step0;
retain complete optimizer/RNG state. Lock both exports before fresh scoring.

## Fresh evaluation and screen

24 entirely new disjoint texts:12 authored /m b p/ and rounded/wide/open vowel
examples plus12 unused CMU ARCTIC. Four known training-held-out engine/gender
presets yield96 new recordings /384 original/half/boost/quarter cases, each with
independent teacher geometry labels and raw surface statistics. Exclude retained
prior texts including targeted-speech training/evaluation. Presets are previously
studied, not new voices. Freeze visual cases and evaluator before scoring.

Primary promising screen: >=3% original-level dense3D surface RMS gain versus
matched control; no engine-average surface RMS regression; gap RMS no more than
1% worse overall or per engine; contact F1 no more than.002 lower; total well-timed
/m b p/ hits not lower and no category loses more than2. Half/boost mean surface
RMS must not worsen. Historical and inspected feature-cohort gap RMS may worsen
at most2% and F1 at most.01 versus released v4. Report all arms, axis errors,
raw-teacher→fitted-rig surface reference, gap/contact/opening/closure/timing and
engine/gain/voice slices. One fixed seed/loss calibration is a pilot, not a release
claim. Preserve failures and do not tune against this confirmation set.

## Browser checks

Exact same ONNX topology/size and no new inference tensors. Check full native/
WebGPU tracks, repeats, finite values and sampled buffer allocations. Three
settled desktop WebKit timing runs after teacher/training complete, with the
normal James page continuing idle animation. Sample simultaneous WebContent+GPU
physical footprints every20ms during short/long/repeated prerecorded inference.
TTS excluded; not a physical-iPhone/full-pipeline under2GB guarantee. No CDN or
Vercel publication from this experiment without a subsequent user request.

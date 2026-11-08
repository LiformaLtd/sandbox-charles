# Training-only SyncNet pilot

Current v4 finetune-r4 remains the published baseline. This experiment compares
ordinary fine-tuning with the same fine-tuning plus a frozen SyncNetv2 mouth
objective. It does not add a network, renderer, phoneme input, lookahead or state
to the browser. Both exports must retain 331,047 parameters and 2,251,827 bytes.

## Preflight before student updates

The preceding compatibility study found that official frozen SyncNetv2 preferred
correct actual-Three animation to shifted, under-opened and missing-closure
variants: eight clips per corruption in each of two independent sets.

The first implementation attempted an offline differentiable renderer of the
same James geometry. Triangle rasterization failed numerical-gradient checks.
A smooth point renderer eventually passed derivatives, timing and under-opening,
but only4/8 missing-closure comparisons. It is rejected and is **not used for
training**. Its intermediate sources, logs and final results remain in ignored
artifacts; a correct derivative of an unsuitable rendering is insufficient.

The final method evaluates SyncNet on **the actual Three James renderer** in a
local WebKit browser. It uses a stochastic central finite-difference estimate,
not analytic backpropagation through Three. Sixteen random Rademacher directions
perturb the23 active mouth/jaw coefficients independently in each of five25fps
frames, with epsilon0.003 and coefficient clipping. Each direction's score
difference contributes its signed direction; their mean estimates a coefficient
gradient. This is a noisy, finite-scale numerical estimate and can miss subtle
improvements. No claim of exact full-coordinate gradients is made.

The new single-view renderer matches four previous reference frames byte for
byte after cropping. Before training, its estimated descent direction must reduce
actual SyncNet distance on≥6/8 first-batch training examples and≥2/4 per engine.
The fixed test moves at most0.001 in any coefficient. All8/8 pass. These checks
show useful local directions, not general perceptual validity or a quality gain.

Each frame uses the prior face camera, BGR0–255 input, crop[16:176,32:192] resized
to224, frozen mouth-expert preprocessing and original checkpoint. Head, gaze,
brows and blinks are held constant only for expert scoring. The actual demo
continues its ordinary idle animation. The renderer RPC binds localhost only;
no speech or model data leaves the computer.

The fixed scorer alignment is−40ms, without changing labels or playback timing.
Five rendered frames at25fps interpolate from the unchanged60fps emitted crop
using exact quantized sample positions. Cached audio embeddings come from the
matching20-MFCC-frame slice. The numerical coefficient gradient passes through
clipping/interpolation and into MLX by the chain rule. A Torch/MLX backbone
vector-product parity check precedes training. It validates the gradient transfer,
not the stochastic estimator's equality to an exact gradient.

## Matched training

Both arms start at the exact current-v4 hash. Both use all old/new primary data,
historical replay, the same seed37 main-batch RNG, AdamW schedule, clipping,
3,720 updates and shared development selector. Each batch has six windows from
each of eight voices plus sixteen replay windows. The original coefficient,
geometry, velocity, contact and timing terms remain unchanged.

The experimental arm adds one five-frame auxiliary example every four updates,
rotating voices and choosing the first eligible window in that voice's existing
six main-batch examples. Thus this is a sparse auxiliary pilot, not an assertion
that all training frames receive visual supervision. It is matched by student
updates and main examples, not wall time. Ineligible groups skip the auxiliary.

The additional loss is `max(student distance − fitted-teacher distance, 0)`.
For a nonpositive unperturbed excess, the auxiliary gradient is zero. Otherwise
central probes estimate the raw-distance gradient. The teacher image score is
cached per immutable training window. Probe seeds are37000+update, independent
of the main sampler, and change at every auxiliary update.
This stops penalizing a student once it matches the teacher's frozen-expert
score, reducing the incentive to beat the teacher through rendering artifacts.
It still may reward visually wrong trajectories. Its scale is fixed using only
the first training batch: eight voice examples (probe seeds100000+voice index) set mean auxiliary backbone
norm to25% of ordinary-loss norm. It ramps linearly for372 updates. There is
no weight sweep or tuning on development/test results.

Select each arm by shared development macro gap RMS +0.25×(1−contactF1), every
124 updates, including initialization. Lock both selections before fresh scores.
The new confirmation has24 distinct texts×four held-out voice presets (96
original recordings), with separate teacher labels for original/half/boost/quarter
audio. All known prior text manifests are excluded before synthesis.

## Evaluation and release decision

Compare both arms and v4 on lip-gap RMS, contact F1 and uniquely associated
/m,b,p/ hits allowing0–100ms anticipatory closure. Report both engines separately,
gain robustness and historical regressions. The promising screen asks for≥3%
RMS gain over control, no engine regression, F1 drop≤.002, total event hits no
lower and no phone losing more than two hits. Prior-cohort RMS may worsen≤2%
and F1≤.01 versus v4. These criteria can fail even if the auxiliary score improves.

Verify identical exported graph/frontend/constants, MLX/ORT and actual WebGPU
parity; measure warmed WebGPU speed and three desktop WebKit memory trials with
James animated idle, short/long speech and repeats. Prerecorded STA-only tests
exclude TTS and do not prove iPhone full-pipeline peaks. No automatic publication
or replacement of v4 follows this research experiment.

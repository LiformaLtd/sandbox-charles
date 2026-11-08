# Training-only SyncNet results

**No meaningful additional gain. Keep published v4 finetune-r4 unchanged.**
The added loss improves fresh lip-gap RMS only0.0103% over matched ordinary
fine-tuning; bilabial hits are identical and contact F1 is slightly lower.
It fails the predeclared≥3% extra-gain and no-engine-regression screen.
This is a negative result for this specific sparse numerical-gradient recipe,
not evidence that every SyncNet or adapted visual-loss approach is ineffective.

[Protocol](compact-syncnet-loss.md) ·
[Reproduction](../../scripts/research/syncnet_loss/README.md) ·
[Learnings](learnings.md#40-training-only-syncnet-preserve-the-actual-rendering-domain-and-require-a-matched-gain).

## Fresh confirmation and browser comparison

The cohort contains24 previously unused texts×four held-out voice presets:
96 independently synthesized original clips, with separately teacher-labelled
half/boost/quarter audio giving384 cases. Primary scores below use original audio.
Both checkpoint selections locked before scoring. Do not directly compare these
RMS values with earlier experiments' different confirmation cohorts.

| Model | Gap RMS (mm) ↓ | Contact F1 ↑ | /m,b,p/ hits ↑ | WebGPU ms /500ms ↓ | Peak MiB, three trials |
| --- | ---: | ---: | --- | ---: | --- |
| Published v4 | 2.47594 | 0.79904 | 39/65/51 | 0.98 | 528.7, 530.2, 524.2 |
| Matched control | 2.44256 | 0.80281 | 41/67/55 | 0.96 | 532.4, 523.3, 529.1 |
| Training-only SyncNet | 2.44231 | 0.80209 | 41/67/55 | 0.99 | 522.1, 522.8, 531.5 |

All exports have331,047 parameters and2,251,827 bytes (2.25 decimal MB). Browser
figures are M5 Max/64GiB desktop WebKit, using the unchanged custom WebGPU runner.
A block is a one-second input emitting a nominal500ms crop. Timings include its
frontend/inference/readback, after warm-up, not TTS or end-to-end generation.
The three-trial median corresponds to about510×/521×/505× real time for
v4/control/SyncNet. Tiny timing differences do not establish a runtime change.

Memory sums simultaneous WebContent+GPU physical footprint at20ms intervals,
starting with loaded James idle and covering session creation, short/long and
three repeated accelerated prerecorded STA passes. James continues rendering
idle. TTS, initial rig loading, network and browser-parent processes are excluded.
This is not a physical-iPhone or full-TTS-pipeline peak-memory measurement.
Explicit model GPU buffers are identical:1,492,236 bytes in25 buffers.

## Quality interpretation

The SyncNet arm improves1.358% over release, while ordinary fine-tuning already
improves1.348%. Do not attribute the release-relative gain to the added objective.
Contact F1 is.802808 control versus.802087 SyncNet. Well-timed /m,b,p/ hits are
41/67/55 for both arms, compared with39/65/51 for v4. Existing unique-event
association and0–100ms anticipatory closure handling are retained.

Phonon gap RMS changes2.176834→2.175854mm (0.045% better); Supertonic changes
2.708282→2.708760mm (0.018% worse). The24-transcript cluster-bootstrap95%
interval for extra improvement is−0.118% to+0.120% (2,000 resamples, seed837),
conditional on these presets/cohort. It supports neither a meaningful gain nor
a significant degradation. This is one training seed, not a multi-seed result.

| Audio | v4 gap mm | Control gap mm | SyncNet gap mm |
| --- | ---: | ---: | ---: |
| original | 2.47594 | 2.44256 | 2.44231 |
| half | 2.59557 | 2.55524 | 2.55781 |
| boost | 2.60253 | 2.57209 | 2.57111 |
| quarter | 3.01749 | 2.96441 | 2.96858 |

| Prior cohort, original audio | v4 gap mm | Control gap mm | SyncNet gap mm |
| --- | ---: | ---: | ---: |
| surface_regression | 2.46046 | 2.41529 | 2.41859 |
| feature_regression | 2.42607 | 2.38722 | 2.39106 |
| historical | 2.54635 | 2.49198 | 2.49362 |
| recent_regression | 2.46400 | 2.41571 | 2.41724 |

Prior dense3D mouth-surface RMS is2.89763mm v4,2.84445 control and2.84749
SyncNet. This is an existing surface regression cohort, not a new fresh dense
surface benchmark. Prior-cohort guards versus release pass; they do not replace
the failed fresh extra-gain/engine criteria.

## How the training gradient was obtained

The browser Three renderer exposes no analytic training gradient. Offline
triangle attempts failed central-difference checks. A smooth point renderer
passed derivatives, timing and under-opening but only4/8 missing-closure
comparisons; it was rejected without training a student. Its failed sources and
outputs are preserved under ignored renderer-trial folders and final renderer
preflight artifacts. Differentiability alone did not validate that training domain.

The final method renders the actual James rig in local WebKit. Four sampled
compatibility frames match byte for byte. The original frozen SyncNet previously
passed both independent actual-render corruption sets8/8 per corruption. Official
checkpoint SHA256 is
`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`.
Camera, rig, blendshape assets and renderer sources match that compatibility study.
Original BGR0–255/MFCC preprocessing and one fixed−40ms scorer alignment remain;
labels and playback timing do not shift. The RPC binds localhost only.

Sixteen Rademacher central probes at epsilon.003 estimate a gradient on23 active
mouth/jaw coefficients in each of five25fps frames. Probes are clipped to[0,1].
This is a noisy finite-scale estimate, not an exact analytic derivative. A
separate max-.001 coefficient descent check reduces actual distance on all eight
first-batch training examples; these are training examples, not independent test
utterances. The cotangent passes through60→25 interpolation/clipping into MLX.
Backbone Torch/MLX vector-product relative difference is below one part per million.

The auxiliary loss is positive excess distance over the fitted teacher's rendered
distance, with zero auxiliary gradient when the unperturbed student already
matches/beats that reference. Its fixed training-only lambda is.00159890757544,
calibrated on the first eight voice examples to25% of the ordinary backbone
norm, ramped over372 updates. Probe seeds100000+voice index calibrate the scale;
training probes use37000+step, independently of the main sampler.

## Matched optimization and limitations

Both arms start at current v4 and use the same primary/replay data, seed37,
AdamW schedule, gradient clipping,3,720 updates and main-batch order. Each batch
contains six windows from each of eight voices plus sixteen replay windows.
The original coefficient/geometry/velocity/contact/timing terms stay unchanged.
One existing eligible primary window receives the auxiliary every fourth update,
rotating voices. Of930 scheduled examples,454 have positive excess distance.
Thus only sparse five-frame clips receive expert gradients, not the full corpus.
The training cost is approximately52s control versus722s SyncNet on this M5 Max;
the browser pays none of the rendering/expert/probe cost.

Both arms select step2976 by the same development metric, including initialization
as a candidate. The control exactly reproduces the previous ordinary-control hash.
Mean preclip norms are1.15025 control /1.18647 SyncNet; clipping fractions are
68.63% /71.29%. These are consequences of the changed objective, not proof of a
specific cause of the null result. This pilot does not isolate estimator noise,
sparsity, expert sensitivity, loss scale or optimization as the root cause.
No post-test tuning, checkpoint changes or publication follow this failed screen.

Useful local score descent and conspicuous-corruption discrimination did not
produce a meaningful held-out student improvement here. A denser or lower-variance
visual objective would require its own frozen matched experiment.

## Verification and retained artifacts

- Identical ONNX operators and frozen frontend/normalization/interpolation/channel
  tensors; identical parameter count and file size.
- Full-waveform MLX/ORT coefficient maxima below9e-7 for both new exports.
- Three actual WebGPU parity trials, short tails, both-engine gain clips and the
  earlier packed-phrase regression; finite complete tracks and no zeroed frames.
- Reset repeats byte-identical; the same1,492,236 explicit GPU bytes for all models.
- Three physical-footprint trials per model with idle draws continuing; fresh-session
  startup medians8.60/8.66/8.68ms for v4/control/SyncNet on local cached assets.
- Sixteen synchronized teacher/v4/control/SyncNet cases pass audio playback and
  exact snapshot-seek checks; sampled review screenshots inspected.
- Frontend build and research Python compilation pass. No production source,
  model default, CDN asset or Vercel deployment changes.

The local review is `/.local/compact-syncnet-v1/review.html` on the research dev
server. Models, predictions, audio, full logs, source locks and physical-footprint
samples remain ignored under `.local/compact-syncnet-v1`; no large weights enter Git.
Selections are `control/selection.json` and `syncnet/selection.json`.

| Model | SHA256 |
| --- | --- |
| Current v4 | `ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da` |
| Control | `63ddc0c64851edc445eba1364e8b160a8a680c4d3b66638fc57a17b782cfe867` |
| SyncNet | `7e0f5992228379af3bc42b010eef62bc7b3c0ad50ec3d653dad14adfa024c6d9` |

Training-protocol SHA256:
`99a0059e7b879ae266deca223ab506b662374aa5d47510a70b2e5d15b02611f8`.

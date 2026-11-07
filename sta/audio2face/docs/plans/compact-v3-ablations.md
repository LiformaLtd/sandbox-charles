# Compact v3: independent width, context and data experiments

Recorded 2026-11-07. This is a controlled fine-tuning pilot, preserving the released
compact v3 and the existing demo. Quality is agreement with the full Audio2Face
teacher; no blinded perceptual study or physical iPhone benchmark is claimed.

## Design

Every arm starts from the same released v3 weights, SHA-256
`738b1a497650f0ffaea698f0d39dd1e78ac81b9c080dfd0cf6155fe365b89df8`.

| Arm | Change from v3 |
| --- | --- |
| Released baseline | No additional training |
| Training control | Six additional epochs, unchanged network and corpus |
| Width 128 | Hidden channels 96 → 128; corpus/context unchanged |
| Width 192 | Hidden channels 96 → 192; corpus/context unchanged |
| Context 1.5 s | Input 16,000 → 24,000 samples; width/corpus unchanged |
| Context 1.5 s + dilation | Same long window, with receptive field 81 → 143 mel frames |
| Extra data | Add 61 new, training-only Phonon clips; width/context unchanged |

All trained arms receive 81 updates per epoch for six epochs: 486 AdamW updates,
learning rate 0.00006 with cosine decay to one tenth, weight decay 0.015,
gradient-norm clipping at 1, seed 37, and batches of 48 new-corpus plus 16 legacy
windows. The final incomplete new-corpus batch wraps to maintain 64 samples.
All use the v3 coefficient/gap/velocity/contact/timing losses and the same
validation selection score. Epoch zero is eligible if fine-tuning never improves
the validation score. Test scores are not used to select epochs.

The control distinguishes a useful change from the benefit of simply continuing
training. Larger initializations have their own RNG consumption; a separate RNG
holds data permutations fixed across width/context arms. Legacy replay order also
remains fixed for the extra-data arm. Added data replaces some exposures within
the same update budget, rather than adding extra optimizer updates.

Width expansion duplicates channels, dividing downstream input weights by their
replication counts. This preserves the starting function. Seeded weight noise of
1e-5 breaks exact channel symmetry. The wider models are warm-started adaptations,
not independently trained-from-scratch capacity comparisons.

The context arm retains the eight dilations `[1,2,4,8,16,1,2,4]`. It supplies 151
mel frames instead of 101, predicts 90 coefficient frames instead of 60, and
supervises frames 30–59 instead of 15–44. Output timestamps and the 8,000-sample
hop are unchanged. This supplies real audio where the short-window model had
padded intermediate context; it does not extend the network's roughly 0.8-second
receptive field. It requires 250 ms more source-audio lookahead during streaming.
A separate context arm also changes block 5 from dilation 1 to 32, increasing
the feature receptive field from 81 to 143 frames (roughly 1.4 seconds) at the
same parameter count. Its initial function changes, so it must adapt the reused
weights during fine-tuning; this is not function-preserving like width expansion.

## Data and evaluation

The 61 additional clips total 203.04 seconds. They were chosen from unused
standalone subtitle candidates, ranked using released v3 errors on training clips
only and subtitle pair frequencies. Every existing train/dev/test transcript,
including the earlier 96 extra clips, was excluded by normalized-text comparison.
Speech uses the existing pinned Phonon Easton workflow. The full teacher and the
same lip-aware James fitter supply coefficients and raw geometry gaps.

The fixed tests are the newer 65 clips, their 29 subtitle and 24 fresh subsets,
the prior-v2 60 clips and 12 conversational subset, and the original 51 clips.
We report raw-teacher lip-gap RMSE, closure precision/recall/F1, velocity error,
coefficient/mouth MAE and automatically aligned `/m/`, `/b/`, `/p/` contact events.
A 2,000-draw paired bootstrap resamples whole clips, preserving temporal correlation
within a clip. It measures test-sample uncertainty, not training-seed variability.

## Browser method

An isolated copy of the current demo uses the same Phonon/James/idle pipeline,
with a research adaptation of the dedicated WebGPU runner for each candidate's
width and temporal dimensions. The frontend, dispatch counts, shader constants,
buffer shapes, output interpolation and streaming crop are changed together.
The active demo and its default model are unchanged.

The tests run in desktop macOS Playwright WebKit 2388 with iPhone 13 viewport/user
agent. A separate browser verifies each candidate against ONNX CPU outputs on
silence, deterministic random input and speech. It does not warm the memory runs.
Every measured cell uses a fresh browser, prepares both sessions, generates and
plays a short or longer clip, and verifies continued idle mesh updates afterwards.

The OS sampler sums simultaneous WebContent and GPU physical footprints every
20 ms. Model buffer bytes are measured separately through the WebGPU API. There
are two fresh-browser repetitions per model/clip cell. These are desktop process
measurements; the dev-build harness also differs from the earlier frozen Vercel
build, so compare candidates within this experiment.

## Reproduction and retained artifacts

The [training and evaluation scripts](../../scripts/research/compact_ablation/README.md)
record the recipe and required corpus layout. Full teacher generation and target
fitting are documented in [learnings.md](learnings.md).

The experiment's models, training histories, predictions, corpus manifests,
parity fixtures and raw memory samples are retained locally under
`.local/compact-ablation/`. Weights and generated corpora are excluded from Git.
Exact historical reproduction still requires the source corpus artifacts and
compatible environments. The report's measured models must be identified by hash.

## Quality results

These are frozen held-out test results. Lower RMSE and higher F1 are better.
F1 is frame-level agreement with teacher contact, not a perceptual-quality score.

| Candidate | ONNX MB | New 65: gap RMSE | New 65: F1 | Prior 60: RMSE / F1 | Original 51: RMSE / F1 |
| --- | ---: | ---: | ---: | --- | --- |
| Released v3 | 2.252 | 0.2246 | 0.8794 | 0.1841 / 0.8562 | 0.3227 / 0.6217 |
| Extra-training control | 2.252 | 0.2212 | 0.8816 | 0.1819 / 0.8566 | 0.3227 / 0.6217 |
| Width 128 | 3.218 | 0.2202 | 0.8818 | 0.1811 / 0.8582 | 0.3228 / 0.6231 |
| Width 192 | 5.935 | 0.2190 | 0.8823 | 0.1805 / 0.8584 | 0.3231 / 0.6241 |
| Longer window only | 2.252 | 0.2210 | 0.8821 | 0.1819 / 0.8557 | 0.3228 / 0.6209 |
| Longer window + receptive field | 2.252 | 0.2213 | 0.8797 | 0.1821 / 0.8563 | 0.3195 / 0.6196 |
| 61 extra training clips | 2.252 | 0.2203 | 0.8801 | 0.1810 / 0.8577 | 0.3225 / 0.6205 |

The 192-channel model reduces newer-test gap RMSE by about 2.5% against released
v3, but only about 1.0% against the matched extra-training control. The latter is
the estimate of the width change's benefit within this training budget. Its
closure F1 gain over the control is 0.0007, or 0.07 percentage points.

The control, width-128 and extra-data models detect 27/33 `/m/`, 22/29 `/b/` and
23/31 `/p/` teacher-confirmed events. Width 192 detects 27, 22 and 24 respectively.
Released v3 detects 27, 22 and 22. The larger-receptive-field arm detects 26, 20
and 22, a regression on these event counts despite lower original-set gap error.
Timing errors are retained in the JSON results; comparisons on different sets
of successful events should not be presented as paired timing improvements.

The paired clip-bootstrap 95% intervals for newer-test RMSE difference relative
to the extra-training control are:

| Change | RMSE difference interval; negative favors candidate |
| --- | --- |
| Width 128 | [-0.00139, -0.00044] |
| Width 192 | [-0.00285, -0.00137] |
| Longer window only | [-0.00059, 0.00047] |
| Longer window + receptive field | [-0.00251, 0.00288] |
| 61 extra training clips | [-0.00146, -0.00030] |

Width and the selected extra data show small RMSE gains in this sample. Neither
context intervention establishes a newer-test gap improvement over the control.
Every width/context F1 interval includes zero; the extra-data F1 interval is
also compatible with zero and leans toward regression. These are exploratory,
uncorrected intervals across several comparisons, with one training seed.
They do not establish a reproducible perceptual advantage.

The width candidates selected epoch six, the end of the fixed budget. This pilot
does not establish their fully converged performance or rule out gains from a
longer schedule. A longer-training study must extend the control too. The extra
data experiment tests 203 seconds of selected speech at fixed updates; it does
not show that larger or more varied datasets are ineffective.

## Checkpoint identities

| Candidate | Learned parameters | Selected epoch | SHA-256 |
| --- | ---: | ---: | --- |
| Released v3 | 331,047 | Released | `738b1a497650f0ffaea698f0d39dd1e78ac81b9c080dfd0cf6155fe365b89df8` |
| Extra-training control | 331,047 | 5 | `36d3391de34faf14b23275060dc6a984f33bdef6beafa0d64ccd8363a90252b1` |
| Width 128 | 572,455 | 6 | `8aa7373ff758c1b3e9386611120ccfca93da11ebbeba4516ace94275645ff9ea` |
| Width 192 | 1,251,879 | 6 | `be654b9dfb4993a426e11751ac74aa46423db9e11ee18238d5fce35267947cde` |
| Longer window only | 331,047 | 5 | `af5d52905c40ae778d17167fc225b3824c20ee186c99eafd820036206106e0c2` |
| Longer window + receptive field | 331,047 | 6 | `b38421a785a1b6cc81116dd3653bb6539201983700092532590122a2de77624e` |
| 61 extra training clips | 331,047 | 5 | `6104627907ee82092ffa259848b3a3debccb30f4bbfdd6eb959735a29471b489` |

## Browser results

All 28 measured runs completed playback and continued idle mesh updates. The
machine identifies itself as Apple M5 Max, 64 GiB unified memory, macOS 26.7.1.
The demo source snapshot was commit `ed4afb3a3805794e122a990cbf995764ba5da2d3`.
Each row below contains the median of two fresh-browser runs for each clip length.
The short and long clips were approximately 4.2 and 12.7 seconds.

| Candidate | Short / long peak MB | Short / long steady block ms | Short / long click-to-audio ms | Tracked model GPU MB |
| --- | --- | --- | --- | ---: |
| Released v3 | 903.8 / 905.5 | 3.04 / 3.81 | 118.0 / 127.0 | 1.492 |
| Extra-training control | 902.1 / 908.0 | 3.13 / 2.80 | 114.5 / 125.5 | 1.492 |
| Width 128 | 895.9 / 915.0 | 5.97 / 3.62 | 117.5 / 124.5 | 2.497 |
| Width 192 | 908.7 / 910.5 | 4.66 / 5.05 | 122.0 / 127.0 | 5.292 |
| Longer window only | 892.2 / 897.9 | 6.48 / 3.23 | 148.5 / 173.0 | 1.575 |
| Longer window + receptive field | 895.5 / 903.7 | 3.37 / 4.68 | 157.5 / 156.5 | 1.575 |
| 61 extra training clips | 903.6 / 899.2 | 3.46 / 3.36 | 114.5 / 122.5 | 1.492 |

The largest individual measured peak was 931.8 MB. Each model
used two compute pipelines and 25 explicit model buffers. Wider weights grew the
tracked GPU buffer capacity from 1.49 MB to 5.29 MB, while the full-demo process
peak remained dominated by other allocations. Small differences between model
process peaks are within the observed variation and do not establish a memory
saving from any candidate.

The wider candidates have a small quality benefit and higher inference cost.
Short-clip timings for width 128 and the window-only arm varied substantially
between two runs, so these measurements do not establish a precise ranking of
those configurations. Warm block times exclude the first dispatch; first-block
values remain in the raw logs. Blocks advance 0.5 seconds of output audio.

The context arms increased click-to-audio time in this setup. Their additional
250 ms of source-audio lookahead is separate from wall-clock TTS latency because
Phonon can generate faster than real time.

Candidate browser outputs agreed with ONNX CPU inference within `5e-5` on all
fixtures. Maximum observed absolute errors were:

| Candidate | Maximum coefficient error | Values compared |
| --- | ---: | ---: |
| Extra-training control | 2.68e-07 | 9,360 |
| 61 extra training clips | 3.28e-07 | 9,360 |
| Width 128 | 3.13e-07 | 9,360 |
| Width 192 | 2.24e-07 | 9,360 |
| Longer window only | 3.43e-07 | 14,040 |
| Longer window + receptive field | 2.24e-07 | 14,040 |

These are desktop results, including a desktop operating system and driver.
They do not prove that an iPhone will use the same memory or inference time.
No candidate has been promoted to the live demo.

## Decision

Width 192 is the strongest newer-corpus numerical candidate in this pilot, with
about 1% lower gap RMSE than matched extra training and one extra `/p/` event.
It does not recover any additional `/m/` or `/b/` events on this small test.
The file grows to 5.94 MB and block inference costs more. Width 128 has a smaller
gain, the context arms have mixed results, and 61 extra clips yield a slight gap
improvement without better closure F1 than the training control.

Keep released v3 as the baseline. A sensible next study would repeat the width
comparison across seeds and extend its training budget alongside the control,
then inspect synchronized mouth close-ups. A human perceptual win and robust
physical-device behavior remain promotion criteria. These runs demonstrate
modest capacity gains, not a major visual-quality upgrade.

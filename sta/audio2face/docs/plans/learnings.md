# Building a small, accurate audio-to-face model with low browser memory

Research notes from the James Audio2Face experiments, consolidated on 2026-11-07 through the matched-seed and channel-folding follow-up.

This document gives a reconstruction procedure for researchers starting with the full teacher and changing the data, fitting objective, student, or browser implementation. Numerical results below describe specific experiments. Hyperparameters are reference settings to repeat or ablate, rather than established optima.

## 1. What we built

The final compact pipeline is:

```text
Phonon speech → 16 kHz mono PCM → log-mel features
                                      ↓
                        small temporal convolution network
                                      ↓
                     39 active weights in a 52-channel vector
                                      ↓
                         authored James blendshape targets
                                      ↓
                           animated mesh and audio playback
```

Full Audio2Face supplies offline training labels. Browser inference uses the distilled student. The deployed compact v2/v3 path computes the fixed audio frontend with a JavaScript FFT and runs the learned network through two reusable WebGPU compute pipelines. James rendering uses Three.js/WebGL, independently of WebGPU inference. Idle head motion and blinking continue during these measurements.

Three decisions produced the main gains:

- Predicting rig controls directly made the model much smaller than compressing the geometry-producing teacher.
- Better training coverage and a lip-aware geometry-to-rig fit improved accuracy at fixed student size.
- A dedicated WebGPU runner removed a large memory spike that persisted when the same small model ran through ONNX Runtime WebGPU.

### Model names

There are two experimental families with overlapping version names. Keep them separate in reports and filenames.

| Family | Outputs | Approximate download | Purpose |
| --- | --- | ---: | --- |
| Full teacher, Multi v3.2 | Geometry, jaw and eye values | 724.84 MB | Label generation and reference |
| Geometry compression v1/v2 and precision/closure variants | Same geometry interface | 72–74 MB | Preserve the teacher interface with quantization and low-rank approximation |
| Compact ARKit v1/v2/v3 | 52 blendshape coefficients, 39 active | 2.25 MB | Direct audio-to-rig distillation |

The compact models have 331,047 learned parameters and 2,251,828-byte ONNX files. The difference between learned parameter bytes and file size includes the fixed waveform frontend and graph data. Compact v2 and v3 use float32 weights. Their size reduction comes primarily from the smaller architecture and output representation.

## 2. Pin and validate the teacher first

The source bundle is named `Audio2Face-3D-v3.0`. The graph used here is identified by the offline implementation as **Multi v3.2**. Record the graph hash rather than inferring its identity from the bundle name. Upstream architecture and usage information are in the [NVIDIA model card](https://huggingface.co/nvidia/Audio2Face-3D-v3.0).

Our reference graph:

| Property | Value |
| --- | --- |
| File | `network.onnx` |
| Bytes | `724844664` |
| SHA-256 | `db47c2701ca849de443c9e9f25657210f829a74fc458ee6fed603a8a501253a8` |
| Identity order | Claire, James, Mark |
| James identity input | `[[0, 1, 0]]` |
| Audio | float32 mono, 16,000 Hz |
| Animation | 60 fps |
| Window | 16,000 samples |
| Hop | 8,000 samples |
| Recurrent state | `[2, 2, 1, 256]` |
| Noise | `[1, 3, 60, 88831]` |
| Emotion input for these experiments | zeros, `[1, 30, 10]` |

A versioned mirror is available at the [full teacher CDN location](https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/geometry-v1/network.onnx). Check the SHA-256 after downloading. The model and the Maya rig assets have associated terms and attribution files. Preserve those with derived artifacts. See the [model source](https://huggingface.co/nvidia/Audio2Face-3D-v3.0) and [James Maya sample](https://github.com/NVIDIA/Maya-ACE/tree/release/2.0/sample_project).

The graph's 88,831 output values per frame consist of:

| Component | Values |
| --- | ---: |
| Skin, 24,002 vertices × xyz | 72,006 |
| Tongue | 16,806 |
| Five jaw points × xyz | 15 |
| Eye values | 4 |

These geometry values need a rig-fitting stage to obtain ARKit coefficients. Appending a 52-value projection after the full graph leaves the upstream network and intermediates in the inference path. Our small student learns audio-to-coefficient prediction directly.

### Timing and state are part of the target definition

Generate one continuous track per clip. Reset state between clips, retain it between windows, and advance it through the initial padded window even when that window emits no valid frames.

The following pseudocode describes the teacher schedule. `pcm` contains the complete clip, and `infer` returns `prediction` and `output_latents`.

```python
state = zeros((2, 2, 1, 256), dtype=float32)
noise = fixed_noise_for_this_experiment
indices = arange(15, 45)
offsets = indices * 16000 // 60

for start in range(-16000, len(pcm) - 4000, 8000):
    window = zeros((1, 16000), dtype=float32)
    left = max(start, 0)
    right = min(start + 16000, len(pcm))
    window[0, left - start:right - start] = pcm[left:right]
    prediction, state = infer(
        window=window,
        input_latents=state,
        noise=noise,
        identity=[[0, 1, 0]],
        emotion=zeros((1, 30, 10), dtype=float32),
    )
    timestamps = start + offsets
    valid = (timestamps >= 0) & (timestamps < len(pcm))
    emit(timestamps[valid], prediction[0, indices[valid]])
```

Assert that emitted sample indices equal `arange(frame_count) * 16000 // 60`, where `frame_count = ceil(samples * 60 / 16000)`. This catches window shifts and dropped boundary frames before training.

The dictionary/v3 teacher references use NumPy `default_rng(0).standard_normal(..., dtype=float32)` for the fixed noise tensor. Early compact v1/v2 labels reused a saved browser-noise tensor. Exact historical replay of those labels requires that tensor. A new experiment can choose its own deterministic seed and regenerate all references consistently.

The MLX teacher path was validated against the source model. When reproducing the dictionary labels, set `MLX_ENABLE_TF32=0` before importing the backend. Omitting that setting produced differences up to about 0.005 in a reference lip-gap probe. With matching settings, the existing reference passed `atol=2e-4, rtol=1e-4`. Check a known clip before generating a corpus.

## 3. Lessons from the 72–74 MB geometry path

We initially compressed the full geometry model using low-rank output/embedding matrices, 4-bit matrix weights, and selective higher precision. The retained geometry interface made direct comparison straightforward, but it kept the recurrent/diffusion machinery and large tensors.

Successful geometry exports included these results on the same 27-clip Chrome evaluation:

| Geometry export | MB | Lip-gap RMS | Closure F1 | Aligned `/m/` contacts |
| --- | ---: | ---: | ---: | ---: |
| Geometry v1 | 72.16 | 0.0754 | 0.942 | 117/206 |
| Geometry v2 | 72.28 | 0.0730 | 0.954 | 111/206 |
| Encoder precision candidate | 73.83 | 0.0698 | 0.955 | 116/206 |
| Encoder plus closure correction | 73.92 | 0.0741 | 0.949 | 153/206 |

Those closure scores use the earlier geometry evaluator and corpus. The frame-level compact-student F1 scores later in this document use a different evaluation protocol. Report each protocol with its numbers.

What this taught us:

- Restore precision one component at a time. Jaw or GRU precision did not provide the best accuracy-per-byte improvement in these tests. Audio-encoder precision was more useful.
- Use a high-precision diagnostic reference to separate quantization error from low-rank approximation error. Restoring precision cannot restore discarded rank.
- Retest combinations. Layer interactions and recurrent feedback can change the result after individually useful modifications are combined.
- Evaluate actual exported graphs in the intended browser. Offline native changes and browser behavior can differ.
- Optimize the perceptual failure being measured. The closure correction caught more `/m/` contacts while aggregate RMS and closure F1 moved in the opposite direction.
- Stronger global smoothing can erase short closures and release edges. Use timing and contact losses to preserve them.

An early residual analysis also found skew, heavy tails and pose-dependent error. In that older geometry compression experiment, the median absolute skew was about 0.78 for jaw outputs and 0.28 for skin coordinates. Jaw median excess kurtosis was about 1.84. These statistics justify measuring conditional errors and tails alongside averages. They describe that geometry residual dataset, with limited generality to a new compact model.

A relative error such as 18.4% requires its denominator and coordinate definition. It does not translate directly into a visual-quality percentage.

## 4. Separate fitting error from prediction error

A compact coefficient model has two sources of approximation:

```text
full teacher geometry → fitted rig coefficients → coefficients predicted from audio
                             fitting error              student error
```

Measure both. An excellent audio predictor cannot recover shapes unavailable in its rig basis. A poor fit also gives the student poor coefficient targets.

For compact v1 on the original 51-clip test set:

| Output | Lip-gap RMSE against full geometry | Contact F1 |
| --- | ---: | ---: |
| Frame-by-frame fitted rig | 0.1424 | 0.7423 |
| Audio student | 0.3470 | 0.5719 |

This showed substantial scope to improve the predictor. The later lip-aware fitter improved the fit itself. On the new v3 65-clip test set, fitted-rig gap RMSE was 0.0350 and F1 was 0.9715, while the v3 student reached 0.2246 and 0.8794.

Fit quality is an attainable reference for coefficient-target reproduction. Contact F1 alone is not a strict ceiling when training also uses full-geometry gap targets. A student can deviate from fitted coefficients and improve that one contact metric.

### Use the rig's exact channel order

Read pose names and active-pose flags from the James model assets. All compact exports emit 52 values. This experiment trains the 39 active channels and maps the other 13 to zero.

The inactive channels are the eight eye-look directions, `jawLeft`, `jawRight`, `mouthLeft`, `mouthRight`, and `tongueOut`. This limits the motions the student can express. The file's 52-value shape alone does not establish support for every ARKit motion.

The student and targets here are James-specific, with neutral emotion conditioning. Other identities, rigs, languages, and emotional control need their own evaluation and usually additional training targets.

### Lip-aware geometry fitting

The current labeler is in the sibling dictionary app. It supplies the channel order, the nine lip landmarks, the bounded solver and raw geometry references. See [label.py](../../../audio2face-dict/scripts/label.py).

Use the source skin neutral `S`, rig neutral `R`, active blendshape basis `B`, and teacher skin displacement `D`. Account for the neutral difference when forming the target displacement, `D + S - R`.

The fitting objective used for the newer `.lip.arkit` targets is:

```text
minimize over 0 <= w <= 1

  || W^(1/2) (B w - (D + S - R)) ||²
  + ||w||²
  + 192 ||r + w C - g_teacher||²
```

`B` is restricted to the rig's frontal mask. Mouth-region coordinates receive weight 4, other coordinates weight 1. `C` contains the active blendshape deltas for nine vertical lip separations, `r` is the rig-neutral lip separation, and `g_teacher` is the full geometry's separation. The ridge coefficient is 1. The lip term was absent from the early label fitter.

The mouth-region mask in the reference fitter is `abs(x) < 3`, `166.1 < y < 169`, and `z > 5` in James source coordinates. These thresholds, the neutral alignment, and the lip weight require revision for another rig or coordinate scale.

The solver forms the 39 × 39 Gram matrix once, uses the inverse to initialize a clipped solution, and performs 95 bounded FISTA iterations with a step of `1 / largest_eigenvalue(Gram)`.

The nine upper/lower skin-vertex pairs are:

```text
(9013,10613), (941,10433), (8859,10434),
(2963,3726), (1439,1475), (5980,7505),
(6093,18810), (263,18809), (17362,13358)
```

Save the full teacher's gaps separately from reconstructed fitted gaps. This preserves an independent geometric reference when the coefficient fitting method changes.

## 5. Use data coverage and model errors together

V1 used 219 training clips and 117,061 valid frames. The corpus included synthetic and natural speech, with only five browser-Phonon training clips. A student intended to animate Phonon should see substantially more of that engine's acoustic characteristics.

V2 added 180 Phonon training clips, about 456 seconds. The new set emphasized `/m/`, `/b/`, `/p/`, word position, neighbouring vowels and ordinary sentences. It had 28 validation clips and 60 held-out clips, including 12 conversational clips. Target words and prompts were separated across its new splits.

V3 combined a larger dictionary corpus with model-error-guided additions:

| New dictionary-derived split | Clips | Audio seconds |
| --- | ---: | ---: |
| Train, including 199 subtitle clips | 453 | 1309.8 |
| Validation | 29 | 91.8 |
| Test, including 29 subtitle clips | 65 | 176.6 |
| Additional new training-only Phonon clips | 96 | 339.4 |

The 453 training clips comprise 239 full-dictionary clips, 15 targeted clips, and 199 subtitle clips. The 65 test clips include 12 full-dictionary tests, 24 fresh clips, and 29 subtitle clips. Counts in earlier discussion were approximate; these are the manifests used for the experiment. V3 also replays legacy compact v1/v2 training windows.

### What OpenSubtitles contributed

The dictionary analysis scanned 83,896,608 subtitle lines. Its stress-free pronunciation dictionary counted 1,440 observed phoneme pairs. A frequency-aware selection produced 250 additional prompts, which became 849.84 seconds of generated and aligned Phonon speech.

Adding their verified pair coverage to the existing dictionary raised estimated occurrence-weighted coverage from 81.49% to 99.733% in that text corpus. This measures the frequency mass of covered phoneme pairs. Timing, amplitude and multiple acoustic realizations still need training examples. See the [analysis implementation](../../../audio2face-dict/scripts/analyze-subtitles.mjs) and [audio verification](../../../audio2face-dict/scripts/verify-subtitles.mjs).

For compact v3, we reused those audio/label pairs and also selected 96 new prompts from the analysis's separate standalone selection. We excluded all previous training, validation and test transcripts. Pair errors were estimated on the training subset only, with at least eight occurrences per pair.

The additional-prompt score was proportional to:

```text
sum over distinct pairs in the candidate
  (pair_gap_RMSE² + 0.20 * pair_contact_error)
  * log(1 + subtitle_pair_frequency)
  / (1 + number_of_selected_prompts_already_containing_pair)

then divide by sqrt(number_of_pair_tokens_in_candidate)
```

This emphasizes measured errors while encouraging variety and repeated evidence for useful transitions. The reference candidate pool also required known pronunciations and excluded words longer than 13 characters.

The 96 new clips used the same Phonon Easton assets, a synthesis temperature of 0.7, and a deterministic seed from the text's SHA-256. Their phone timings were aligned with the dictionary project's TinyAligner path. Preserve the exact TTS package, weights, voice, tokenizer and resampling implementation, especially when comparing against browser speech.

### Split before examining errors

Normalize transcripts across projects and voices, then exclude exact normalized overlap between training and evaluation. Keep repeated voices for an identical prompt in one split. Hashing the transcript makes the assignment independent of prompt-selection order.

The reference normalization is `" ".join(re.findall(r"[a-z0-9]+", text.lower()))`. The v3 subtitle split uses the first eight hexadecimal digits of SHA-256 of that normalized transcript, modulo 10. Bucket 0 is test, 1 is validation, and the remaining buckets are training. Existing dictionary test/validation sets retain their roles, and the fresh set is evaluation-only. New error-guided clips are training-only.

Exact text deduplication does not test semantic novelty. A broader study should also hold out speakers, synthesis engines, speaking styles and languages. Preserve the older evaluation sets while adding new ones so gains on a newly chosen corpus remain assessable.

### Cache labels with provenance

A useful per-clip record contains:

```json
{
  "id": "stable-clip-id",
  "text": "The spoken transcript.",
  "split": "train",
  "samples": 48000,
  "audio_sha256": "...",
  "teacher_model_sha256": "...",
  "teacher_identity": "James",
  "teacher_noise_seed": 0,
  "fit": "lip-gap-192",
  "channel_order_sha256": "...",
  "phones": [{"phone": "M", "start": 0.1, "end": 0.2}]
}
```

The shared dataset files are headerless little-endian float32:

| Suffix | Shape or interpretation |
| --- | --- |
| `.pcm` | 16 kHz mono samples |
| `.lip.arkit` | `[ceil(samples * 60 / 16000), 52]` fitted coefficients |
| `.geometry-gaps` | `[frames, 9]` raw teacher gap references |

Validate byte lengths, finite values, coefficient range, channel order, and audio hashes before reuse. A label file should be invalidated when its audio or target-generation settings change.

## 6. Reference student architecture

The compact network operates on one second of waveform and predicts 60 coefficient frames. Training and streaming use the valid centre 30 frames.

### Audio frontend

| Setting | Value |
| --- | --- |
| Sample rate | 16,000 Hz |
| Window | Symmetric Hann, 400 samples, `numpy.hanning(400)` |
| FFT | 512 points |
| Hop | 160 samples, 10 ms |
| Padding | 200 zeros on both sides |
| Spectrum | Power, 257 nonnegative-frequency bins |
| Mel bank | 64 triangular bands, 60–7900 Hz |
| Compression | `log1p(10 * mel_power)` |
| Feature shape | `[batch, 64, 101]` |

Use the same mel definition as the export. The original generator used `2595 * log10(1 + hz / 700)` to place band edges uniformly in mel space, linear triangle slopes in Hz, and no extra band normalization. Mixing library defaults silently changes these features.

The ONNX waveform export represents the windowed DFT as a convolution with 514 real/imaginary filters. The dedicated runtime replaces this fixed computation with an FFT and reads the original window/mel values from the model. Both paths need numerical parity checks.

### Learned network

```text
64 input channels
  → Conv1d(64, 96, kernel=5, padding=2)
  → GELU, tanh approximation
  → 8 residual blocks, dilations [1,2,4,8,16,1,2,4]
  → sample/interpolate feature positions at 60 fps
  → Conv1d(96, 39, kernel=1)
  → transpose to [batch, 60, 39]
  → scale and mean
  → clamp to [0,1]
  → fixed mapping to [batch, 60, 52]
```

A residual block is:

```python
x + 0.5 * conv1x1(gelu(conv3_dilated(x), approximate="tanh"))
```

The 3-tap convolution has padding equal to dilation. The 1×1 convolution mixes channels. Initialize normalization statistics from training labels only, with per-channel `mean` and `scale = max(std, 0.06)`. V2/v3 retain v1's normalization buffers during fine-tuning.

Frame `f` samples feature position `(floor(f * 16000 / 60)) / 160`, interpolating between its floor and successor. Preserve the integer floor before the division. The browser projection shader implements that ordering explicitly.

The student has local temporal context and no carried GRU state. It needs lookahead supplied by the overlapping windows. The first useful streaming block requires roughly half a second of source audio. A prepared-model click-to-audio benchmark also includes TTS generation and buffering, and has a different interpretation from file inference time.

### Training window construction

For the student, start windows at sample -8000 and advance by 8000. Use the same zero padding, frame positions 15 through 44, and validity mask as the teacher's emitted track. Crop the prediction to these centre frames for the loss. Exclude padded frames from every loss and metric. Exclude invalid adjacent pairs from velocity losses.

## 7. Losses that helped

Let `p` be predicted active coefficients, `y` fitted target coefficients, `s` the frozen per-channel scale, and `g` the nine raw teacher gaps. Reconstruct predicted gaps from clamped coefficients with the rig basis.

The v2/v3 objective is:

```text
L = L_coeff + 1.2 L_gap + 0.12 L_velocity
    + 0.22 L_contact + 0.45 L_contact_timing
```

Definitions:

- `L_coeff` is Smooth L1 on `(p - y) / s`, with beta 0.5. Mouth and jaw channels have weight 1.7, other active channels weight 1. Normalize by valid frames × 39.
- `L_gap` is Smooth L1 between reconstructed and raw gaps, beta 0.14. Normalize by valid frames × 9.
- `L_velocity` is the absolute difference between adjacent predicted and target coefficient changes, divided by `s`. Normalize by valid-frame count × 39 after masking invalid pairs.
- Contact is `median(nine_gaps) <= 0.10`. Use BCE-with-logits with prediction logit `(0.10 - median(predicted_gaps)) / 0.16`. Positive contact frames receive weight 2, other frames weight 1.
- `L_contact_timing` is Smooth L1, beta 0.10, between adjacent changes of the median gap. Include valid adjacent pairs when either teacher median gap is below 0.35. Normalize by the number of selected pairs.

The 0.10, 0.14, 0.16 and 0.35 constants are in the source rig's coordinate units. Rescaling a rig requires corresponding changes.

V1 used the coefficient, gap and velocity terms. The contact and edge-timing terms helped address missed closures without relying solely on globally smoothing the track. Extra pressure for positive contacts can increase recall while reducing precision. Retain both in evaluation.

### Optimization settings

A fresh v1-style training reference uses seed 37, width 96, batch size 64, AdamW, weight decay 0.015, gradient-norm clipping at 1, and a 28-epoch one-cycle schedule with maximum learning rate 0.002 and warm-up fraction 0.12. Select by validation loss. These are the original trainer defaults.

The recorded fine-tunes used:

| Experiment | Initialization | Epochs | Initial LR | Batch composition | Selected checkpoint |
| --- | --- | ---: | ---: | --- | --- |
| Compact v2 | v1 | 12 | 0.00018 | 32 new + 32 legacy windows | Epoch 8 |
| Compact v3 stage A | v2 | 12 | 0.00012 | 48 new + 16 legacy windows | Epoch 10 |
| Compact v3 stage B | Stage A winner | 6 | 0.00006 | Same mix, with 96 extra clips included | Epoch 5 |

Fine-tuning uses AdamW with weight decay 0.015, gradient clipping at 1, and cosine LR decay to one tenth of the initial LR. Batches are shuffled with a deterministic seed. Legacy replay is sampled each epoch rather than requiring every old window in each epoch.

For v3, select checkpoints by:

```text
F1_new + 0.60 F1_prior_v2 + 0.25 F1_original
- 0.30 RMSE_new - 0.20 RMSE_prior_v2 - 0.10 RMSE_original
```

All terms in that selection score come from validation splits. The weighting is a design choice used in this experiment. Record changes to it before evaluating a new test set.

An early MPS fine-tuning attempt collapsed in this environment, including a lower-LR retry. The successful v2/v3 fine-tunes ran on CPU. The cause of the MPS failure remains unresolved. Compare losses and gradients on a CPU reference before relying on a different training backend.

## 8. Evaluate visible behavior with several metrics

Our compact-model measurements are agreement with the full teacher, supplemented by rig/contact inspection. A blinded perceptual study remains outstanding.

The compact evaluator uses:

- Lip-gap RMSE over all nine landmarks and valid frames.
- Frame contact precision, recall and F1 using median gap ≤ 0.10.
- Gap-velocity error for opening/closing dynamics.
- Coefficient MAE over active channels, with mouth/jaw MAE reported separately.
- Automatically aligned `/m/`, `/b/`, `/p/` events whose teacher contains a contact lasting at least two frames.
- Contact onset and release errors. Report whether timing statistics use all hits or only events detected by both candidates.

For phone-event association, the reference evaluator matches a teacher contact against a phone interval with 45 ms of leading allowance and 35 ms of trailing allowance. Predicted contacts are matched one-to-one by temporal overlap. An automatically aligned event is a proxy label, with its own alignment error.

Also inspect a synchronized mouth close-up. Look for closures, lower-lip/teeth interaction, lip roll, corners, jaw motion, popping at window joins, and the transition to idle. A high aggregate coefficient score can hide a missed `/m/`, and a model that holds the lips shut too often can improve recall while looking worse.

### Measured compact-model progression

On the original 51-clip test set:

| Student | Gap RMSE | Contact F1 |
| --- | ---: | ---: |
| v1 | 0.3470 | 0.5719 |
| v2 | 0.3315 | 0.5998 |
| v3 | 0.3227 | 0.6217 |

On the new v3 evaluations:

| Held-out set | Clips | v2 F1 | v3 F1 | v2 gap RMSE | v3 gap RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined new test | 65 | 0.8406 | 0.8794 | 0.2686 | 0.2246 |
| Subtitle subset | 29 | 0.8499 | 0.8996 | 0.2887 | 0.2346 |
| Fresh subset | 24 | 0.8560 | 0.8885 | 0.2539 | 0.2160 |
| Prior v2 test | 60 | 0.8651 | 0.8562 | 0.1948 | 0.1841 |
| Prior v2 conversational subset | 12 | 0.8084 | 0.8307 | 0.2540 | 0.2171 |

V3 reduces new-test gap RMSE by 16.4% and subtitle-subset RMSE by 18.7%. Its prior v2 contact F1 falls slightly while gap error improves. On that prior set, contact precision drops from 0.8400 to 0.8052 and recall rises from 0.8917 to 0.9142. Preserve this tradeoff in any claim that v3 is better.

On the new 65-clip set, teacher-confirmed contacts detected were:

| Phone | Eligible events | v2 detected | v3 detected |
| --- | ---: | ---: | ---: |
| `/m/` | 33 | 26 | 27 |
| `/b/` | 29 | 20 | 22 |
| `/p/` | 31 | 22 | 22 |

A good dataset addition can improve general transitions even when one small event count stays flat.

## 9. Measure memory as a separate objective

A 2.25 MB model caused roughly 2 GB of full-demo WebKit peak memory through one runtime path. File size is an inadequate estimate of process memory.

Keep separate measurements for:

1. Downloads and cache copies.
2. Model preparation, parsing and weight upload.
3. Prepared idle sessions.
4. First inference and shader compilation.
5. Steady streaming inference and audio playback.
6. End-of-clip cleanup and settled memory.
7. Second and later generations with different text.

A projected 52-value output from the full graph can still retain large intermediates. For scale, the full graph's explicit float32 diffusion-noise tensor alone is about 64 MB. The compact graph avoids that machinery.

### Our measurement method

The benchmark machine was the user's 64 GB Apple Silicon MacBook Pro, described as M5 Max. Tests used macOS Playwright WebKit with an iPhone 13 viewport/user agent and fresh browser contexts. Hardware and OS allocations remained those of the Mac.

The sampler used `proc_pid_rusage(..., RUSAGE_INFO_V4, ...)` at 20 ms intervals. It recorded `ri_phys_footprint` and resident size for the new WebContent and GPU processes. Reported totals sum physical footprints at the same sample instant. Process attribution can include platform/shared costs, and this sum is an operational comparison measure.

Do not sum each process's separate maximum at different timestamps. Also distinguish the median of per-run peaks from the largest observed peak.

We additionally instrumented `GPUDevice.createBuffer`, `GPUBuffer.destroy`, compute pipeline creation, queue submission and audio-source start. GPU-buffer bytes describe explicit buffers observed through that API. Browser/framework allocation, driver resources and process heaps require process-level measurement.

A 2 GB budget was a working iPhone target motivated by crashes. A universal Safari tab limit was not established. Physical-device browser, OS, foreground/background state and available memory remain part of the final acceptance test.

### Isolated full-demo breakdown

One 4.2-second desktop WebKit comparison gave:

| Scenario | Peak MB |
| --- | ---: |
| Idle page with live James renderer | About 390 |
| Phonon plus playback, animation inference bypassed | 731 |
| Compact v2 CPU plus prerecorded audio | 565 |
| Phonon + compact v2 CPU | 817 |
| Phonon + compact v2 ONNX Runtime WebGPU | 1976 |

Subtracting idle suggests about 340 MB for the Phonon path and about 170 MB for the isolated compact CPU path. These are separate runs with shared overhead and different allocation lifetimes. Their differences are approximate attributions rather than additive model-memory measurements.

The local DAWN worker was measured separately at about 2.57 GB. That process belongs to the optional local head-motion backend. The hosted demo uses a precomputed motion track, so its client-memory measurements exclude that worker. Keep server-side and client-side memory budgets separate.

The user found CPU animation too slow on the iPhone. WebGPU remained a requirement even though desktop CPU had a lower memory peak.

## 10. What caused the WebGPU spike

The controlled tests localised the spike to the ONNX Runtime WebGPU execution path. Exact internal allocator attribution remains unmeasured.

The following interventions helped distinguish likely causes:

| Experiment | Observation |
| --- | --- |
| Skip compact inference but leave sessions prepared | Full-demo playback peak fell to about 1.06 GB |
| Reuse GPU buffers with `simple`/`bucket` cache modes | Allocation churn fell, while peak remained around 2 GB |
| Move only fixed audio preprocessing to a JS FFT | Peak remained around 2 GB |
| Change constant folding behavior | No convincing peak reduction in this test |
| Replace the runtime path with a dedicated WebGPU runner | Full-demo peaks fell to about 0.83–0.85 GB |

In an instrumented 4.2-second run, the original lazy-release path created about 1,483 buffers and allocated about 72 MB cumulatively. The simple cache created about 190 buffers and allocated about 8 MB cumulatively. Peak process memory barely changed. Cumulative GPU allocation and retained process memory were different problems.

One disabled-cache configuration failed playback. Aggressively releasing buffers needs functional verification alongside memory checks.

The instrumented general-runtime path created 23 compute pipelines. The dedicated runner uses two pipelines with 25 reusable buffers and about 1.49 MB of tracked GPU model-buffer capacity. Most of the earlier spike was charged to the WebContent process, while explicit live GPU buffers were only a few MB. These observations support the runtime-path change without proving that one particular compiler or allocator caused the whole spike.

ONNX Runtime remains useful for model interchange, CPU reference inference and other model/runtime combinations. Rebenchmark newer versions against the same exported model. Runtime behavior can change. See its [WebGPU documentation](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).

## 11. The dedicated WebGPU runner

The implemented runner reads float initializers from the same ONNX files. It preserves the trained weights while implementing a fixed topology in WGSL. Relevant sources are [compact-webgpu.js](../../src/compact-webgpu.js), [compact-mel.js](../../src/compact-mel.js), [convolution kernel](../../src/compact-conv.wgsl), and [projection kernel](../../src/compact-project.wgsl).

Its reusable resources are:

| Resource | Shape or role |
| --- | --- |
| Input features | `64 × 101`, float32 |
| Hidden states | Three `96 × 101` float32 buffers |
| Weights and output metadata | Packed float32 parameters, means, scales and channel indices |
| Output | `60 × 52`, float32 |
| Readback | Same output size, reused after unmapping |
| Uniform buffers | Fixed metadata for each layer and the projection |

For each one-second audio window:

1. Compute the log-mel features with a reusable FFT workspace.
2. Upload features into the existing input buffer.
3. Encode 17 convolution dispatches and one projection dispatch in one compute pass.
4. Copy the output to the reusable readback buffer, submit, map, copy the small result, then unmap.
5. Emit valid centre-frame coefficients and play the corresponding audio.

The convolution kernel supports the encoder, dilated convolutions and residual channel-mixing layers. It fuses GELU where appropriate and performs the residual addition in the mix layers. It uses workgroups of 16 × 8. The projection kernel interpolates 100 Hz features to the 60 fps frame positions, applies mean/scale and clipping, and maps the 39 active channels into 52 outputs. Its workgroups are 8 × 8.

The FFT implementation uses float64 CPU accumulators and float32 features. The model kernels use float32. Numerical agreement, rather than an assumed identical accumulation order, validates the replacement. Buffer usage, mapping and device-loss behavior follow the [WebGPU specification](https://www.w3.org/TR/webgpu/).

### Verify numerical parity before judging quality

For v2, 43,680 output values differed from ONNX CPU inference by at most about `4.62e-7`. For v3, 46,800 values differed by at most about `4.47e-7`. Fixtures included silence, deterministic random input and held-out speech windows.

For your variant, compare the full 52-channel outputs over multiple windows, including boundaries and silence. Check streaming track lengths and time positions too. A new width, activation, normalization, channel map or temporal schedule requires corresponding runtime changes.

The current reader checks expected float-array lengths and channel mapping. It is a specialised reader for these exports, with topology supplied by the kernels. Researchers changing graph semantics must validate those semantics explicitly even if tensor names and sizes still match.

### Measured runtime comparison

The first integrated runner experiment reduced the 4.2-second full-demo peak from about 1.98 GB to 0.83 GB and the 12.7-second peak from about 2.16 GB to 0.84 GB.

A later paired v2/v3 comparison through the dedicated runner gave the following medians of two fresh-context runs per cell:

| Model and clip | Peak MB | Median steady block ms | Click-to-first-audio ms |
| --- | ---: | ---: | ---: |
| v2, 4.2 s | 816.8 | 3.56 | 120 |
| v3, 4.2 s | 835.8 | 3.28 | 118 |
| v2, 12.7 s | 836.8 | 3.22 | 127 |
| v3, 12.7 s | 845.5 | 3.39 | 131 |

Both models had exactly 1,492,236 bytes of tracked GPU model buffers. Small process/time differences across two repeats should be treated as variability requiring more runs if a small regression matters. These results establish similar desktop behavior with live idle animation. A physical iPhone comparison is a separate measurement.

## 12. Preparation, streaming and repeated-play lifetimes

The prepare buttons were introduced to diagnose which model startup triggered an iPhone crash. Preparing both should leave both sessions usable for Generate & Play. Repeated worker startup can add latency and another allocation peak.

The implemented and tested lifecycle choices include:

- Serialize cold model startup on mobile to avoid overlapping preparation peaks. Keep both prepared sessions afterwards.
- Retain one TTS worker and one animation session. Reset a failed loader promise so a later attempt can retry.
- On mobile, disable the Phonon SDK cache path that returned model bytes to the page and copied them into a cache response while the worker was live. HTTP caching and SDK caching have different allocation behavior.
- Release the application's downloaded model backing buffer after parsing/upload or ONNX session creation. `ArrayBuffer.transfer(0)` is used when available. Runtime-owned weight storage remains live for inference.
- Reuse GPU weights, activations, bind groups and readback memory. Unmap before copying into the readback buffer again.
- Dispose ONNX inputs and outputs on both normal and exceptional paths. Release old recurrent-state tensors in the geometry path.
- Resume the AudioContext during the button gesture so asynchronous model work can lead to audible playback on mobile.
- Use the playback clock to select animation frames, including seek/replay. Keep sample indices as the common timing basis.
- Store the completed audio/animation clip for replay. On Apple mobile with the same text and engine, the Generate button reuses that clip. Edited text must still be tested as a new generation.
- Reload when changing model/runtime to reclaim the old runtime context. Restore the dialogue and comparison settings.

In a desktop repeat test, the first generation had the highest peak and later generations declined. The user had reported a second-play iPhone crash. That combination supports explicit repeated-play tests, while leaving the physical crash mechanism unresolved.

A test that terminated Phonon after every synthesis worsened the second-run peak and added roughly 2.2 seconds of restart latency in that setup. Keeping the worker warm was more useful for the intended streaming interaction.

The current windowing/replay implementation still accumulates audio and animation data proportional to utterance length. Very long or unbounded streaming needs a separate bounded ring-buffer design, plus a decision about replay retention. The 12.7-second tests do not establish unbounded-stream memory stability. See [audio windowing](../../src/geometry.mjs), [playback](../../src/stream-playback.js), and [session lifecycle](../../src/inference.js).

## 13. Preserve the avatar behavior while optimizing

The product requirement is that James continues blinking and moving his head while idle. Freezing him was rejected despite its memory savings. Keep that requirement in performance tests.

An idle-only desktop WebKit investigation found:

| Rendering configuration | Approximate steady MB |
| --- | ---: |
| Continuous idle, current settings | 362–363 |
| Continuous idle, 10–30 fps | 358–363 |
| 1× render resolution with antialiasing | 341–342 |
| 1× resolution without antialiasing | 332–335 |
| Hold a static rendered frame | 152–153 after settling |

Lowering idle frame rate did little for steady memory in this test. Holding a static frame, or drawing only around blinks, let memory settle between bursts but changed the avatar's behavior and did not eliminate burst peaks. These remain diagnostic controls rather than the deployed idle policy.

The rig and packed blendshape downloads total about 5.4 MB. Rendering/driver overhead was much larger than those file sizes. Lower resolution can give a modest memory saving with a visible quality tradeoff. The main speech-time saving came from changing the inference runtime.

Keep authored eyelid closure, procedural blink timing, head-motion smoothing, neck/shoulder deformation and eye counter-rotation separate from the audio student's accuracy assessment. The hosted demo's precomputed head track and procedural blinks contribute visible behavior that the compact student does not generate. Full local DAWN gestures incur a separate server budget. See [rig.js](../../src/rig.js) and [idle-motion.js](../../src/idle-motion.js).

## 14. A procedure for a fresh research variant

1. **Pin a baseline.** Record the full graph, rig assets, channel order, package versions, precision settings, deterministic seeds and hashes. Verify one teacher clip and its timestamps.
2. **Create independent splits.** Include natural speech from the deployment TTS, phonetic contrasts and a broader speech set. Reserve conversational and difficult-transition tests before analysis.
3. **Generate full teacher references.** Preserve recurrent state and window alignment. Save raw geometry gaps along with coefficient labels.
4. **Validate the representation.** Compare a frame-by-frame rig fit against the full geometry before training audio prediction. Inspect mouth close-ups and fix the fitter when it is the main source of error.
5. **Train the small reference student from scratch.** Start with the architecture/frontend and losses above. Choose checkpoints by validation data and retain an unchanged baseline export.
6. **Add data based on training errors.** Use pronunciation coverage and frequency statistics to select varied sentences for weak transitions. Avoid promoting test-set errors into training selection.
7. **Fine-tune with replay.** Keep general speech examples in the batch. Evaluate contact precision/recall, timing, gap geometry and active coefficients across every frozen test set.
8. **Verify the browser implementation.** Check full-output numerical parity against ONNX CPU inference, then run the complete TTS → animation → rig → playback path.
9. **Measure lifecycle peaks.** Test both prepare orders, cold Generate, first inference, long clips, cancellations, changed-text repeats and model/runtime switching. Keep live idle motion active.
10. **Measure the actual deployment device.** Record iPhone model, OS, browser and foreground conditions. Treat desktop emulation as a development comparison.
11. **Publish reproducible artifacts.** Include data provenance, splitting rules, training code, fitting code, checkpoints, hashes, metric definitions, browser build and raw timing/memory traces with each result.

For controlled variants, change one factor at a time, then combine successful changes and retest. A useful experiment matrix separates data volume, label fitting, loss weights, network capacity, weight precision and runtime implementation.

Open research directions include broader natural speech, more realizations per frequent transition, longer temporal context, modest width increases, rig-specific corrective shapes for residual fitting error, and quantization of the already small student. Each requires fresh parity, quality, speed and memory checks. The observations here do not establish that a wider or lower-precision student will improve the final tradeoff.

## 15. Artifacts, source map and reproduction boundary

Published compact ONNX files contain the learned and fixed float32 tensors needed to inspect or initialize a variant. The following hashes identify the exact measured releases:

| Release | SHA-256 |
| --- | --- |
| Compact v1 | `572573e2b32c0fe6ed5b6ce953cc71151619db10f78673b0a89ead0fc696212a` |
| Compact v2 | `ba31af50b3052ed5e65dcb216a230ed91e29a737dad159d4f8ca59b76df882c3` |
| Compact v3 | `738b1a497650f0ffaea698f0d39dd1e78ac81b9c080dfd0cf6155fe365b89df8` |

Versioned files, byte counts, checksums, notices and licenses are listed in the [v1 manifest](https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/arkit-student-v1/manifest.json), [v2 manifest](https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/arkit-student-v2/manifest.json), and [v3 manifest](https://staging-cdn.liforma.ai/models/speech-to-animation/lf-sta/arkit-student-v3/manifest.json).

All three ONNX files are 2,251,828 bytes. The v1 rig artifact `james-blendshapes.bin` is shared by the compact versions. V2/v3 retained the graph topology, shapes, fixed frontend and normalization buffers, replacing only 36 learned weight/bias tensors. ONNX export used opset 17.

Relevant implementation paths, relative to this document:

| Concern | Source |
| --- | --- |
| Model/runtime selection and cleanup | [src/inference.js](../../src/inference.js) |
| Audio-window timestamps | [src/geometry.mjs](../../src/geometry.mjs) |
| Dedicated GPU runner | [src/compact-webgpu.js](../../src/compact-webgpu.js) |
| Audio frontend | [src/compact-mel.js](../../src/compact-mel.js) |
| WGSL kernels | [convolution](../../src/compact-conv.wgsl), [projection](../../src/compact-project.wgsl) |
| Blendshape application | [src/blendshape-rig.js](../../src/blendshape-rig.js) |
| Packed rig preparation | [scripts/prepare_blendshapes.py](../../scripts/prepare_blendshapes.py) |
| Phonon worker lifecycle | [src/phonon.js](../../src/phonon.js) |
| Label fitting in the sibling project | [scripts/label.py](../../../audio2face-dict/scripts/label.py) |
| Subtitle selection | [scripts/analyze-subtitles.mjs](../../../audio2face-dict/scripts/analyze-subtitles.mjs) |
| Subtitle audio verification | [scripts/verify-subtitles.mjs](../../../audio2face-dict/scripts/verify-subtitles.mjs) |

The browser snapshot used `onnxruntime-web` 1.31.1, `phonon-tts` 0.6.0 and `protobufjs` 7.6.6. `package-lock.json` records the installed build dependencies. Phonon asset revisions and model URLs are pinned in [vercel.json](../../vercel.json).

Historical trainers and raw benchmarks were executed in a research workspace outside this application checkout. The research folders were named `arkit-student`, `arkit-student-v2` and `arkit-student-v3`. Their scripts include `train.py`, `labels.py`/`label.py`, `score.py`/`evaluate.py`, the v3 corpus/reference/extra-data generators, numerical parity tests and OS memory samplers. The complete workspace and corpus are not bundled by these CDN model manifests.

On the development machine, selected v3 manifests and results are retained under the ignored `.local/compact-v3/` directory. Local files are supplementary evidence, not portable dependencies for a fresh clone. Exact historical score reproduction needs an archive of the original audio, labels, cached noise, split manifests, training scripts, environment lockfiles and raw benchmark traces. The recipe above supports a new experiment from the pinned teacher with independently generated data; its measured scores should be reported as a new run.

This document is an experimental record and reconstruction recipe. It intentionally retains regressions, rejected approaches and measurement limits alongside the successful changes so a new researcher can choose a different objective and verify the consequences.


## 16. Isolate capacity, context and data from extra training

The [first controlled ablation study](compact-v3-ablations.md) compared widths
128/192, longer input context, a larger temporal receptive field, and 61 extra
training clips. Every trained arm had the same six-epoch, 486-update budget.
An unchanged 96-channel model trained for the same budget was the control.
Without that control, most of the benefit would have been attributed to the
intervention rather than simply continuing optimization.

On the existing 65-clip test set:

| Intervention | ONNX MB | Gap RMSE | Contact F1 |
| --- | ---: | ---: | ---: |
| Released v3 | 2.252 | 0.2246 | 0.8794 |
| Additional training only | 2.252 | 0.2212 | 0.8816 |
| Width 128 | 3.218 | 0.2202 | 0.8818 |
| Width 192 | 5.935 | 0.2190 | 0.8823 |
| Longer input window only | 2.252 | 0.2210 | 0.8821 |
| Longer window and larger receptive field | 2.252 | 0.2213 | 0.8797 |
| 61 extra targeted clips | 2.252 | 0.2203 | 0.8801 |

Width 192 improved gap error by only about 1% beyond the training control.
The extra 61 clips, totaling 203.04 seconds, produced a small gap improvement
without better closure F1 than the control. They were selected using training
errors and subtitle frequencies, excluding every historical transcript. This
result tests one data addition at fixed updates; it does not establish that
more data or greater acoustic diversity would be ineffective.

Distinguish an input window from a receptive field. Extending the waveform from
one to 1.5 seconds supplied 151 mel frames instead of 101, while retaining the
same eight dilations and roughly 0.8-second feature receptive field. This
replaces some boundary padding with real audio. A separate arm changed block 5
from dilation 1 to 32, extending the feature receptive field from 81 to 143
frames. Neither produced a clear newer-test gain over the control. The latter
also missed more labial contacts, despite improving gap error on the original
corpus. Both needed 250 ms more source-audio lookahead.

These are warm-started, fixed-budget fine-tunes. They do not compare fully
converged architectures trained independently from scratch. Wider networks were
initialized by duplicating channels and dividing downstream fan-in weights by
the replication counts, preserving the initial function apart from 1e-5 weight
noise. Retest a different initialization or training schedule as a new experiment.

## 17. Repeat across seeds and reserve a new confirmation set

The [longer follow-up](compact-v3-width-followup.md) ran all three widths for
12 epochs at seeds 37, 73 and 101: nine fits, each with 972 optimizer updates.
Every update contained 48 frozen v3-corpus windows and 16 legacy replay windows.
The 61-clip pilot addition was excluded to keep the width comparison separate.
The cosine LR schedule ran from 0.00006 to 0.000006 over the full 12 epochs.

A longer schedule should be specified explicitly. This was a new 12-epoch run
from released v3, not a restart of a decayed six-epoch optimizer with an
undocumented LR reset. Checkpoints used the same validation objective, with
epoch zero still eligible. Some runs selected earlier checkpoints; one wider
run improved again at epoch 11 after several flat epochs. That observation
argues against assuming that an early plateau proves convergence.

Use separate random generators for initialization and sample order. Otherwise,
a wider initializer consumes more random draws and silently changes the batches
even when the nominal seed is identical. Within each seed pair, the study held
window and legacy replay order fixed. Independent fine-tuning seeds estimate
variation around a fixed v3 initialization, not variability across training an
entire student from scratch.

Repeated comparisons on the same test set can overstate generalization. Before
evaluating the follow-up, we froze 48 newly written prompts: eight each for
conversation, `/m/`, `/b/`, `/p/`, connected speech, and pauses. Exact normalized
transcripts did not overlap any historical split or the previous data addition.
The resulting 256 seconds contained 15,379 frames and 287 eligible teacher
contacts. These are new transcripts from the same Phonon Easton distribution,
not held-out speakers or synthesis engines. Labels remained automatic, using
the full teacher and the same lip-aware fitter.

The following are means across three seeds; released v3 is one fixed reference:

| Model | Fresh 48: gap RMSE | Fresh 48: contact F1 | Original 51: gap RMSE / F1 |
| --- | ---: | ---: | --- |
| Released v3 | 0.2580 | 0.7618 | 0.3227 / 0.6217 |
| Longer-trained 96 | 0.2548 | 0.7691 | 0.3226 / 0.6221 |
| Longer-trained 128 | 0.2539 | 0.7700 | 0.3242 / 0.6204 |
| Longer-trained 192 | 0.2517 | 0.7728 | 0.3248 / 0.6173 |

Width 192 reduced fresh-set gap error in every matched seed: about 1.24% beyond
the longer-trained control and 2.45% against released v3. A crossed paired
bootstrap resampled seeds and whole clips together, preserving within-clip
correlation. Its 95% interval for RMSE difference against the control was
approximately [-0.00495, -0.00143]. The F1 interval, [-0.00025, 0.00804], included
zero. Three seeds remain a limited estimate of training variability.

The original diverse corpus regressed slightly. Width 128's fresh-set gains
were inconsistent across seeds. Report these outcomes alongside the gains;
they do not support claiming a universal improvement over v3. Representatives
for browser and visual checks were chosen by validation, not by test scores.

Compare timing on the same detected events. Pooling matched seed-events, the
192-channel intervention improved `/m/` and `/p/` release MAE by approximately
2.5 and 4.4 ms, while their onset MAE worsened by approximately 0.9 and 2.9 ms.
It did not improve every aspect of lip synchronization. Averages over different
successful-event subsets can conceal that tradeoff.

## 18. Train wider, then fold equivalent replicas back to the small graph

Inspect whether a larger trained network actually uses distinct extra features.
Our 192-channel warm starts retained almost identical channel replicas; encoder
replica cosine similarity was approximately 1.0. This led to a secondary,
deterministic transformation that used the trained weights, without fitting to
confirmation examples.

For the modulo-96 duplication used here, merge the 192 channels into 96:

1. Average encoder output-channel replicas and their biases.
2. In every hidden convolution, sum weights across duplicated input channels,
   then average across duplicated output channels. Average the hidden biases.
3. Sum duplicated input columns in the final projection, preserving its output
   biases. Keep normalization, channel mapping and audio/timestamp buffers fixed.

In notation, for hidden convolution weights:

```text
W_fold[o, i, k] = mean over output replicas a of
                   (sum over input replicas b of W_wide[a, b, k])
b_fold[o] = mean over output replicas a of b_wide[a]
```

This is algebraically equivalent when corresponding replica features are
identical. With learned divergence it can be lossy; it is not generic pruning
or a way to halve any independently trained network. The current helper requires
an integer multiple of 96 channels in the original replica ordering; it does not
implement the nonuniform duplication case used for width 128.

Validate the operator sequence, convolution attributes, frontend, normalization,
channel map and temporal buffers before exporting. Then compare full outputs on
silence, deterministic random input and speech, followed by complete held-out
tracks. The [folding helper](../../scripts/research/compact_ablation/fold_width.py)
performs structural checks; these do not replace numerical or quality evaluation.

For our three seeds, all 224 historical/confirmation clips per seed were checked:
**8,638,344 coefficients** in total. The largest absolute folding difference was
**8.94e-7**, with mean absolute differences around 1.2e-8. All reported rounded
quality scores matched the corresponding wide models. Browser inference of the
selected folded checkpoint matched ONNX CPU outputs within 3.73e-7 across 9,360
additional fixture values.

The resulting model has **331,047 learned parameters** and is exactly
**2,251,828 bytes**, matching the original 96-channel export. It retains the
wider fine-tune's measured function while using the original inference topology.
This demonstrates that a wider fine-tune can alter optimization dynamics without
requiring more deployed parameters. It does not demonstrate a fundamental
capacity limit of a normally trained 96-channel student. A larger model with
independent features, different initialization or more training could behave
differently.

Validation selected width-192 seed 37 at epoch 8 for the representative folded
candidate. Its SHA-256 is:

`b87c52993117fa32d5d48ab97851314996177fd0dd254384596fb898efc52f2f`

It is a local experimental artifact under
`.local/compact-followup/james-arkit-student-folded-candidate.onnx`, not one of
the published v1/v2/v3 releases listed earlier. Keep v3 available while evaluating
the candidate's narrower Phonon-focused gains and diverse-speech regression.

## 19. Count emitted frames when reporting real-time speed

The standard model reads an overlapping one-second input window but emits
**30 usable centre frames**, advancing **0.5 seconds of audio** at 60 fps.
Boundary blocks can be shorter. For steady blocks, the speed factor is:

```text
real-time factor = 500 ms of emitted audio / measured block milliseconds
```

Using the full one-second input duration would overstate streaming throughput
by a factor of two. Keep warm inference, startup, source-audio lookahead and
click-to-first-audio latency separate. Warm measurements here include the audio
frontend, GPU inference and readback, but exclude TTS generation.

The follow-up ran 30 fresh-browser tests: five model choices, two clip lengths
(about 4.2 and 12.7 seconds) and three repeats. Training stopped before timing.
The same Phonon/James pipeline retained continuous idle animation.

| Representative | Short / long block ms | Short / long peak MB | Explicit model GPU MB |
| --- | --- | --- | ---: |
| Released v3 | 3.02 / 3.02 | 906.0 / 899.7 | 1.492 |
| Longer-trained 96 | 2.74 / 3.18 | 895.8 / 897.3 | 1.492 |
| Longer-trained 128 | 3.78 / 3.64 | 896.9 / 902.0 | 2.497 |
| Longer-trained 192 | 4.96 / 4.80 | 909.8 / 910.7 | 5.292 |
| Folded candidate, 96 at inference | 2.98 / 3.04 | 896.8 / 900.3 | 1.492 |

The folded candidate therefore achieved approximately **164–168× real time**,
with the original **1,492,236 bytes** of explicit model GPU buffers. All models
used two compute pipelines and 25 model buffers. Prepared click-to-first-audio
medians for the folded model were 115 and 123 ms. Several short-run timings were
noisy; report repeats and ranges rather than treating a single run as precise.

The largest individual process peak was 913.8 MB. These are desktop macOS
WebKit 2388 measurements on the M5 Max with iPhone 13 viewport/user-agent
emulation. The 20 ms sampler sums simultaneous WebContent and GPU physical
footprints; it does not establish unique total-system memory or physical iPhone
requirements. Small peak differences are within the observed variability.
Model-buffer equivalence is stronger evidence than an apparent few-MB reduction
in whole-browser memory.

## 20. Consistent targets and reliable experiments are further optimization work

A training-label audit found different levels of agreement with raw geometry:

| Training corpus | Clips | Fitted-label gap RMSE | Fitted-label contact F1 |
| --- | ---: | ---: | ---: |
| Legacy v1 | 219 | 0.1367 | 0.7583 |
| Legacy v2 | 180 | 0.1173 | 0.7305 |
| V3 dictionary | 453 | 0.0356 | 0.9721 |
| Earlier 96 extra clips | 96 | 0.0355 | 0.9737 |

These are different corpora, so the table does not isolate the fitter's causal
effect. It does show that legacy replay supplies less faithful coefficient
targets. The next useful data experiment is to regenerate lip-aware targets on
the same legacy training audio while preserving teacher noise, precision and
timing, then compare otherwise identical training. Keep test references frozen.
More data with inconsistent labels can retain a systematic error rather than
remove it. This relabeling experiment has not yet been performed.

Additional practical lessons from the research harness:

- Freeze scripts, input/model hashes, splits and environment versions per study.
  Save optimizer, scheduler, model, best checkpoint and RNG state so interrupted
  work can be reconstructed. Do not silently change the LR schedule on resume.
- Fingerprint audio, labels and manifests when caching features; changing a label
  fitter must invalidate cached targets. Completed-run reuse must also account
  for changed inputs. Write cache files atomically before concurrent readers use
  them. In-process reuse avoids loading the same dataset repeatedly.
- Separate seed variability from clip-sampling uncertainty. Keep older tests as
  regression checks, and choose representative models using validation. A new
  exact-transcript holdout still does not establish unseen-speaker generalization.
- Pre-bundle lazy worker dependencies in a Vite benchmark. The first cold attempt
  reloaded during preparation when Vite discovered `phonon-tts/worker.js`.
  Pre-bundling it fixed the harness; accepted runs assert one main-frame
  navigation. A dev-server reload must not be misdiagnosed as a memory crash.
- Use one waveform, playback clock and seek position for visual comparisons.
  Hold the head steady when inspecting lips, but keep continuous idle animation
  in the separate runtime-memory tests. The teacher-fitted lane is distinct from
  rendering the full teacher mesh. Fixed closure frames looked broadly similar;
  they did not establish a dramatic perceptual upgrade or a blinded preference.

The [research scripts and setup](../../scripts/research/compact_ablation/README.md)
include the model, losses, matched-study driver, original confirmation prompts,
evaluation, folding, browser sampling and synchronized comparison tools. The
[follow-up report](compact-v3-width-followup.md) retains full experiment details.
Raw corpora, frozen historical recipes, checkpoints and traces remain local,
ignored artifacts; exact replay still requires their archive and compatible
teacher/dictionary dependencies. Treat a regenerated corpus as a new experiment.

## 21. Train for the target engines and audit early contact separately

The [multi-voice study](compact-multivoice.md) and
[results](compact-multivoice-results.md) change the research target from mostly
Phonon Easton to male/female voices from Phonon and Supertonic 3. The 336-clip,
16-voice pilot reserves entire voices and texts for validation/test. All clips
have freshly generated, consistently fitted teacher labels. Eight training
voices are sampled equally per optimizer update; no runtime voice ID is needed.

The 336 recordings contain only 52 distinct texts: 32 training prompts across
eight voices, eight validation prompts across four other voices, and 12 test
prompts across four further voices. All 39 English phone types occur in training,
but neither phone-type coverage nor repeated recordings of a small prompt set
establish broad lexical/acoustic coverage. Reserve both entire voices and
transcripts, and report per-engine and male/female preset slices. The headline
multi-voice metric averages voices equally; the older Easton report pools frames.
Do not compare those aggregates as though they were the same test distribution.

At the same 331,047 learned parameters and 2.25 MB export, data-only fine-tuning
reduced unseen-voice macro lip-gap RMS from 3.140 to 2.795 mm. Adding a
training-only phoneme head or a projection to teacher speech features produced
essentially tied test results. This one-seed, 52-text pilot does not establish
that those methods cannot work; it does not establish an additional gain either.
Do not combine auxiliary tasks simply because their underlying ideas are sound.

The full teacher's own last audio-encoder layer can supply offline distillation
features. A copied backend exposing it matched the untouched teacher's geometry
exactly over roughly four million values. Both auxiliary heads are discarded
before ONNX export, so neither TinyAligner nor a large speech encoder is added
to the browser. Checks confirm gradient flow into the existing encoder and the
absence of auxiliary weights from the exported graph.

Broadening data can cause forgetting. The initial pilot worsened older Easton
RMS by 6–7%. A follow-up with 48 new windows plus 16 clean v3/v3-extra replay
windows, and an Easton development-set term, retained the unseen-voice gain and
brought old Easton RMS to 2.597 mm versus v3's 2.580 mm; F1 improved to 0.7696
from 0.7618. Keep both generalization and established-voice checks. That follow-up
was chosen after inspecting the pilot and is not fresh independent confirmation.

A matched-budget Easton-only diagnostic used the same training texts and 744
optimizer updates but sampled only Easton recordings. No trained epoch improved
the unseen-voice validation score, so checkpoint selection retained epoch zero.
This argues against explaining the multi-voice result merely as extra Easton
fine-tuning. It remains a limited diagnostic: only one seed, with different
amounts of unique acoustic data. The clean-replay follow-up also changed the
validation cohort by adding Easton, so it is an engineering recipe rather than
a pure one-factor estimate of replay's benefit.

Closure events do not need to coincide with acoustic phoneme starts. The old
45 ms teacher-event association could omit early contacts. The corrected audit
uses 100 ms pre-padding, reports shared/multiple/truncated associations, and
compares predicted onset/release with the teacher's actual, unshifted geometry.
Keep “any overlap” hits distinct from stricter timing/IoU scores. Do not move all
animation by a presumed 50 ms, count one contact twice for adjacent labials, or
interpret aligner-relative offsets as biological ground truth. The complete
exclusion counts and 45/100/150 ms sensitivity are saved. Existing gap RMS and
framewise contact F1 are unaffected by this association correction.

Make the operational definitions explicit when publishing a hit rate:

- Association uses 100 ms pre-padding and 35 ms post-padding around the acoustic
  phone interval. A contact may begin earlier if it continues into that window.
  These tolerances associate an episode; they do not grant the prediction a
  100 ms timing error or cap physiological anticipation at 100 ms.
- Contact is median separation across nine lip pairs at or below 0.10 cm
  (1 mm). The phoneme-event denominator includes only uniquely associated
  teacher episodes lasting at least two 60fps frames. Absent, shared, multiple
  and clip-truncated reference contacts are reported separately.
- An overlap hit requires any temporal intersection with the actual teacher
  contact, with one-to-one matching. A separate "well-timed" hit requires temporal
  intersection-over-union at least 0.5 and onset error at most two frames
  (33.3 ms). Report signed onset/release errors too. This is a benchmark criterion,
  not a biological correctness threshold.

Under that corrected policy, the replay candidate scored:

| Sound | Unique teacher episodes | v3 overlap hits | Replay overlap hits | v3 / replay well-timed hits |
| --- | ---: | ---: | ---: | ---: |
| `/b/` | 51 | 28 | 42 | 24 / 34 |
| `/p/` | 62 | 29 | 49 | 24 / 39 |
| `/m/` | 43 | 22 | 28 | 19 / 25 |

These counts supersede earlier greedy-association counts; changing the denominator
is an evaluation correction, not an additional model improvement. They measure
agreement with teacher contacts, not success on every acoustic labial. Teacher
onsets in the uniquely associated subset had median offsets near -93 ms for
`/b/` and -100 ms for `/p/` relative to TinyAligner starts, with earlier tails.
Such measurements include CTC boundary uncertainty and coarticulation. Do not
turn an assumed typical 50 ms lead into a global audio/animation correction.

This timing distinction also applies during learning: phoneme auxiliary targets
describe acoustics; the animation targets remain the teacher's original geometry
frames. Do not replace them with closures centred on a phone timestamp. The
student has lookahead and can learn anticipatory articulation.

Do not assume a teacher is perfectly invariant to gain. In a 16-training-clip
diagnostic, the compact student changed its predicted gaps about five to seven
times more than the teacher under half/double gain, suggesting an acoustic
robustness opportunity. But the teacher's largest clip-level change was 0.414 mm,
too large for the predeclared unchanged-label tolerance. Properly regenerate
targets for such augmentation or validate a new normalization method separately.

Desktop WebKit full-page peaks for the replay candidate were approximately
895 MB with Phonon and 1,636 MB with Supertonic 3 on the measured short/long
clips; model GPU buffers remained 1.49 MB and idle animation continued. Those
are summed WebContent/GPU process footprints, not physical-iPhone guarantees.
The TTS engine materially affects the full-page budget even when STA is tiny.

Warm animation blocks remained about 3 ms for 30 emitted frames (0.5 s), with
two compute pipelines and 25 buffers. Replay CPU/WebGPU parity was within
2.69e-7 across 9,360 coefficients. Auxiliary supervision did not increase the
2.25 MB deployment graph. Small measured timing/memory differences between these
identical-shape exports are not established optimizations. The measurements cover
fresh-browser short/long playback, not a repeated-playback soak or physical
iPhone validation; TTS startup and generation are not included in the block time.

Two harness failures also matter for reproduction. NumPy/Accelerate's batched
matrix product emitted floating-point warnings during mel preparation; use the
explicit `einsum` reduction from the controlled ablation, check finite outputs,
and exclude the interrupted diagnostic run. Vite does not implement Vercel's
asset redirects, so serve the pinned Supertonic files explicitly in the isolated
harness. Its first missing-asset attempt failed and is excluded from accepted
memory measurements. Keep those harness failures separate from model quality or
mobile memory failures, and retain the frozen scripts/hashes for completed runs
when extending the trainer with new options.

## 22. Confirm a candidate with new voices and repeated fresh generations

The [v4 validation protocol](compact-v4-validation.md) and
[results](compact-v4-validation-results.md) repeat the clean-replay recipe with
fine-tuning seeds 37/73/101 and add 96 fresh clips: 24 new texts spoken by
Phonon Trevor/Pippa and Supertonic M3/F3. Freeze the protocol and text/voice
definitions before evaluating predictions. Select the representative using only
development scores. Here that selected seed 73, epoch 19, even though seed 101
later had the lowest fresh test error. Do not replace a representative after
looking at test scores.

The fresh mean lip-gap RMS improved **5.4%**, from 3.042 to 2.879 mm, versus the
exploratory pilot's roughly 11%. That smaller confirmation gain is the stronger
estimate to quote. All three seeds improved; their RMS values were 2.897, 2.880
and 2.859 mm. Mean contact F1 improved from 0.7604 to 0.7828. A paired crossed
seed/voice/text bootstrap gave an RMS-difference interval [-0.301, -0.025] mm,
with only three seeds and two new voices per engine. These runs share released
v3 initialization and test fine-tuning randomness, not independent full training
histories. More clips do not remove the uncertainty from having few speakers.

Report slices even when the aggregate passes. Phonon mean RMS improved 7.8%,
Supertonic 2.9%, while F3 alone was effectively tied (+0.1% RMS). Its contact
F1 improved. The result does not mean every voice or every aspect of articulation
improved. Historical Easton mean RMS was within +0.34% of v3 with better F1;
keep such regression checks separate from fresh confirmation.

All predeclared desktop engineering checks passed. The representative remains
2,251,827 bytes and 331,047 learned parameters, with no additional inference
encoder or aligner. Its artifact hash and validation selection are frozen in
the local candidate manifest. Passing those checks supports a v4 candidate for
device review; it does not automatically promote the model or establish full
teacher fidelity. See the results report for the new closure-event counts using
the corrected association and stricter timing definitions.

Repeated-playback tests must actually generate new audio and parameters. On
Apple mobile paths, clicking Generate again with identical text can use the
last-clip replay shortcut, hiding repeated-generation allocations. Prepare once,
then use three different texts on the same page and verify fresh non-silent
audio, fresh model inference, unchanged navigation count, rendered frames and
continued idle animation. Avoid model teardown or forced garbage collection
between plays when testing the intended persistent-session behavior.

In desktop WebKit, representative candidate process peaks across three fresh
generations declined 883 → 858 → 813 MB with Phonon and
1575 → 1547 → 1519 MB with Supertonic. Model GPU allocation stayed exactly
1,492,236 bytes, 25 buffers and two pipelines. This found no cumulative growth
in three plays, not proof of no long-term leak. Sampling began after initial rig
readiness, so it does not certify cold page/rig-loading peaks. Neither this
desktop emulation nor a sub-2-GB process sum replaces physical iPhone testing.

Finally, wait for the **rendered** frame after seeking in scientific comparisons.
An audio seek is asynchronous: a successful snapshot-button click or updated
slider can precede the avatar's actual pose change. The confirmation verifier
waits until the viewer's frame equals the requested teacher contact/opening frame
before taking screenshots. Keep every comparison lane on one audio/frame clock;
do not interpret a screenshot captured before seeking completes as a closure
failure or success.

Hosted downloads need their own end-to-end check. During v4 deployment,
Supertonic worked with local files but WebKit rejected the Vercel → Hugging Face
redirect chain for its voice JSON because of the credentials/CORS transition.
Resolve hosted style/config/model URLs directly to the pinned upstream, using
the shared asset resolver, rather than assuming same-origin redirect paths work
inside workers. Re-test the real deployment with both engines and model switches;
a successful local build or CDN checksum does not establish browser fetchability.

## 23. Establish numerical parity before treating browser defects as model errors

A conspicuous neutral-face snap around “packed” affected both v3 and v4 in the
shared custom WebGPU runner. On desktop WebKit/Metal, large positive arguments
to WGSL `tanh` returned NaN. The projection/clamp turned the invalid hidden
values into all-zero final poses. Thus a check for finite final coefficients
passed even though animation was corrupted. Short silence/random-input parity
fixtures had not exposed the speech-activation peaks.

Bound the GELU approximation's tanh argument to [-10, 10], where float32 tanh
is already saturated. This preserves the trained function, with no added
buffers, pipelines or model weights. Across full browser-generated Phonon and
Supertonic passages and three loudness stress cases, both models now agree
with native CPU to less than 0.000001 over 492,856 coefficients. Before the
fix, the recorded Phonon passage lost one v3 frame and nine v4 frames around
the plosive; the fixed outputs lost none. See
`compact-webgpu-activation-regression.md` for the detailed evidence and harness.

Archive **actual browser PCM** for parity tests. A similarly configured offline
TTS invocation may still generate different audio. Compare every emitted frame
from the complete streaming crop, including sharp consonants and multiple
voices/engines; include amplified audio as a separate numerical stress test.
Inspect suspicious all-zero poses as well as NaNs. Verify an unstable primitive
in isolation, then compare the corrected activation to an independent framework
and the full GPU output to native inference. Fix implementation parity before
changing training targets, capacity or losses in response to browser artifacts.
Desktop WebKit evidence still needs confirmation on the physical target phone.

The repaired hosted build also passed the complete reported paragraph in all
four combinations: v3/v4 with Phonon/Supertonic. Inspecting GPU readbacks found
no zeroed centre-crop frames or page errors in 56 windows per Phonon run and
68 per Supertonic run. Separate checks verified audio playback, the WebGPU
default, model switching and continued idle animation on the public deployment.
Keep these implementation checks separate from teacher-fidelity measurements:
native/GPU agreement establishes that the browser runs the trained compact
model correctly, not that the compact model reproduces the full teacher exactly.

This is a useful low-memory optimization lesson: better numerical behavior can
recover visible quality without increasing model size or runtime allocations.
The shader fix adds no buffers or pipelines, but that observation is not a new
measurement of total page peak memory. Preserve the previous measured memory
results and remeasure when changing allocation lifetimes, topology or renderer
assets; do not infer a new phone memory result from a shader arithmetic change.

## 24. Gain augmentation helps robustness but needs a matched fine-tuning control

The [teacher-labelled gain follow-up](compact-gain-robustness-results.md) tested
original versus original/half/headroom-boosted speech, with a newly generated
teacher track for every recording. Both arms started from v4 and used the same
744 updates, development selection and clean replay. Amplification respected
per-clip peak headroom; quarter-level speech was reserved for evaluation.

Gain augmentation reduced half/boost macro lip-gap RMS by 3.4% versus the matched
control, missing the predeclared 5% gate. Quarter-level error improved 7.3% versus
control, but ordinary-level error improved only 0.3%. Its 1.6% ordinary-level
improvement versus v4 would overstate the gain intervention if the extra-training
control were omitted. Contact improved on level-shifted speech, but the experiment
did not deliver a broad quality breakthrough. Keep v4; do not silently lower the
gate, pick another seed against this test set, or promote an experimental export.

All original-level teacher labels regenerated exactly, while teacher geometry
changed slightly with waveform gain. This confirms both that supervision was
consistent and that unchanged-label augmentation would have been inaccurate.
A source-hash cache guard initially rejected labels after logging-only source
changes. Rather than weaken the guard or mix provenance, the new setup regenerated
all labels under one current recipe. Keep archived recipes and failed-setup records.

The gain export retained 331,047 parameters and 2,251,827 bytes. Robustness data
can therefore help without adding runtime encoders or model memory, but identical
topology is not a new peak-memory measurement. More level variants also do not
add distinct transcripts or articulatory contexts. The larger remaining fidelity
gap still calls for broader speech coverage and independent representation tests.

After reviewing the gain-study results, the user requested publication as an
update to v4. That deployment choice supersedes the initial recommendation to
keep the older weights; the failed 5% study gate and one-seed limitation remain
in the evidence. Publish a new immutable revision and change the demo's model
pointer, preserving the predecessor for rollback. Do not overwrite a cached
model URL or attribute the original v4's three-seed confirmation to a later
single-seed fine-tune. V4 gain-r1 retains the 2.25 MB label and original topology.

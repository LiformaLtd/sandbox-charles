# Targeted difficult speech: fixed pilot

Published v4 finetune-r4 stays unchanged during this experiment. Initialize every
arm from ordinary control SHA ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da.
No teacher-feature head, model capacity, precision or browser-runtime changes.

## Target selection without test leakage

Score current v4 on primary TRAINING windows only, using existing teacher gaps.
Associate per-frame squared error with aligned phoneme triples, including word
boundaries. Keep contexts supported by at least eight distinct normalized texts,
at least100 frames, and occurring in at least three unused candidate sentences.
Rank RMS times sqrt(texts/(texts+8)), selecting four bilabial, eight vowel and
four other centre-phone contexts where available; fill remaining slots by rank.
This shrinkage/support filter discourages isolated noisy examples. Do not mine
confirmation error or choose contexts using newly generated teacher labels.

Exclude every retained prior text, including feature/scale evaluation. Partition
unused supported CMU ARCTIC texts by a fixed hash:80 reserved evaluation
candidates, remainder training candidates. Select72 broad-training texts by an
independent fixed hash first. For each broad text select an unused targeted
text of similar word count (within one if available), maximizing selected-context
score per phone triple. Match voices and gain schedules by text index. Report
actual context enrichment and length differences before optimization.

Both additions contain72 distinct texts × four training presets =288 new
recordings, each with one independent teacher-labelled gain level, balanced
across eight voices and original/half/headroom-boost. Generate unchanged pinned
Phonon/Supertonic settings. This is a fixed context-enrichment intervention,
not a universal active-learning algorithm or a pure phoneme-count experiment.

Reserve32 new evaluation texts from the80-text partition before synthesis:
16 context-enriched (including /m b p/ and vowels) and16 hash-selected broad
sentences. Four known training-held-out presets yield128 fresh recordings /
512 independently teacher-labelled original/half/boost/quarter cases. Never use
these waveforms or labels to select training examples or hyperparameters.
Report enriched and broad evaluation strata separately as well as the combined
macro-voice score. All text is unchanged ARCTIC; retain prior source/license.

## Matched optimization and selection

Three arms: original-data-only control, broad addition, targeted addition.
All use the current expanded data and historical replay, seed37,3720 updates,
AdamW beta.9/.999 eps1e-8 decay.015 clip1, cosine LR.00012→.000012.
For each update draw six existing windows per eight training voices. The two
addition arms replace two of each voice's six windows with new-data windows;
original control retains all six. Clean replay stays16 windows in every arm.
Use separate PCG64 old/replay/new streams seeded1037/2037/3037 so original
window draws and replay draws remain identical across arms. Broad/targeted have
identical new-window proportions; their recordings/windows differ deliberately.
This oversamples the additions; it is not uniform sampling of the union.

Keep the existing animation loss and frontend. Reuse read-only prior caches;
cache novel speech with the same exact window/crop/mask. Use validated MLX FP32,
TF32 disabled. Evaluate initial/every124 steps on unchanged common development
and Easton guard; select macro voice RMS+.25*(1−contact F1), allowing step0.
Lock all three exported checkpoints before scoring fresh speech. Save complete
optimizer/RNG/checkpoint provenance and both old/replay sample hashes. No sweep,
new seed or prompt-selection adjustment using confirmation results.

## Screen and runtime

Primary screen: targeted versus broad ≥3% original-level RMS gain on fresh
combined cohort; no engine-average RMS or contact F1 regression, no broad-stratum
RMS regression, no half/boost mean RMS regression. Well-timed /m b p/ total must
not fall and no category loses more than2. Historical and inspected prior-feature
RMS may worsen at most2% and F1 at most.01 versus released finetune-r4. Report
all arms versus released v4 and original-data control, both evaluation strata,
gain conditions, opening error and onset/release/closure audits. One fixed seed
is a pilot, not broad independent release confirmation. Keep any failed screen.

Verify exact331047 parameters/2251827 bytes and identical ONNX topology, no new
browser inputs. Check complete native/WebGPU tracks and repeat resets. Compare
three settled warm browser runs and20ms sampled WebContent+GPU footprints on the
normal animated James idle page with short/long/repeated prerecorded STA. TTS
is excluded; no physical-iPhone memory guarantee. No automatic model deployment.

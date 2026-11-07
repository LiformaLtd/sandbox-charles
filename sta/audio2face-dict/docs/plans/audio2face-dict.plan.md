# Audio2Face dictionary demo plan

Build a standalone SvelteKit 3 app in `sandbox-charles/sta/audio2face-dict`. Use full Audio2Face v3 offline to create a library of facial animation tracks. In the browser, synthesize speech with Phonon, align its phonemes with full TinyAligner, and animate the James rig by selecting, retiming, and blending tracks from the library.

## 1. Prove the offline pipeline on 20–30 clips

- Synthesize short English lines with Phonon’s Easton voice, the default in the existing studio. Feed the same 16 kHz mono PCM to full Audio2Face v3 and full TinyAligner. Pass the text used to synthesize each clip to TinyAligner as its transcript.
- Use the existing full v3 Apple Silicon runner for offline geometry generation and the existing James ARKit fitting code to produce 52 weights per frame at 60 fps. If the native backend specifically needs to be NVIDIA’s SDK, run the offline exporter on a supported Windows or Linux machine with an NVIDIA GPU. [NVIDIA SDK requirements](https://github.com/NVIDIA/Audio2Face-3D-SDK#system-requirements)
- Verify the channel order against the James blendshape rig and confirm frame zero, frame count, audio duration, and visible lip closure before scaling up. Save the clip ID, transcript, voice, PCM, phoneme labels and start/end times, and coefficient track. Cache synthesis, alignment, and animation separately so failed clips can be retried and the index rebuilt independently.

## 2. Build a corpus for phoneme coverage

Start with several hundred short lines totaling about 10 minutes of speech. Choose prompts that add useful pairs and triples, including across word boundaries. Keep multiple examples of common combinations with different natural timings. Split by normalized transcript before indexing so repeated takes of a sentence stay in the same split. Use a development set to find missing combinations, adjust timing rules, and decide whether to expand to 20 minutes. Reserve a separate test set until the corpus and matching rules are fixed.

## 3. Index all pair and triple occurrences

Walk each clip in phoneme order. Use the gap from one phoneme’s end to the next phoneme’s start to flag a possible pause. Start with a 120 ms gap. TinyAligner’s returned phoneme list omits CTC blanks, which can include audible speech. Confirm a pause when the intervening PCM contains at least 120 ms of consecutive low-energy windows. Measure RMS in 10 ms windows, tune the energy threshold on the development set, and share the same detector between corpus generation and playback. Keep unconfirmed gaps, shorter gaps, and ordinary word boundaries eligible for matching.

A single run with 20 phonemes contributes 19 pair occurrences and 18 triple occurrences. Each ordered pair or triple key maps to all matching occurrences, including repeated matches within a clip. An occurrence stores a clip ID and the first phoneme index. The clip record holds the phoneme timings and the offset and length of its animation track in a shared frame buffer. Index every individual phoneme occurrence using the same clip and phoneme references. Mark occurrences from isolated runs so they can be preferred when animating a one-phoneme run.

Store each animation track once. At 52 Float32 weights and 60 frames per second, the tracks take about 7.5–15 MB for 10–20 minutes of audio, before the index. Start with an in-memory index and typed arrays. Measure browser memory before adding quantization or splitting the bundle. Ship the index and weights to the browser. Keep corpus audio and fitted tracks in the offline cache. Fit raw geometry as it is produced and retain lip landmark measurements from full geometry for the test set.

## 4. Select the nearest triple and fill gaps with pairs

Use each aligned phoneme’s midpoint as its timing anchor in the first version. For each sliding triple in the new speech, search its exact key and score each occurrence by the two intervals around its middle phoneme. With `s` for the stored source and `t` for the new speech, use this score.

```text
((s₂ − s₁) − (t₂ − t₁))² + ((s₃ − s₂) − (t₃ − t₂))²
```

Pick the lowest scoring occurrence and break ties by stable corpus order. After selecting triples, fill each uncovered interval within the run with the best matching pair, scored by the squared difference between its source and target anchor spacing. When neither match exists, try the two single phonemes as a fallback. Use the outgoing half of the first phoneme and the incoming half of the second. Reuse the source phoneme from each neighboring exact match where available, and blend between the two halves across the interval. If either phoneme is absent, count the interval as a miss. Report fallback coverage separately from exact triples and pairs. For a run with one phoneme, choose the closest duration among isolated occurrences of that phoneme if available. Otherwise use the closest duration among its occurrences within longer runs. Count these matches separately from pairs and triples. If the phoneme is absent, use neutral and record a miss.

## 5. Retime and blend the motion

For each output frame, locate its time between target anchors and map that position linearly to the corresponding source interval. Interpolate between the two neighboring source frames to sample the 52 coefficients. At a run’s first and last phonemes, extend a match that covers the edge phoneme by mapping start to midpoint for the first phoneme and midpoint to end for the last. Use both mappings for a single-phoneme match. Keep unmatched edges under the miss rule. Fade to or from neutral in the silence before and after each run where time is available, so the fade leaves the onset and release intact.

For consecutive triples `ABC` and `BCD`, both contribute on `B–C`. Start that interval with 100% `ABC` at `B` and ramp linearly to 100% `BCD` at `C`. Use a single track at full weight wherever there is only one contributor outside a join.

Extend a selected double a short way before its first timing anchor and after its last wherever it joins another matched span. For doubles `AB` and `BC`, keep sampling `AB` just past the midpoint of `B` and start sampling `BC` just before it. Map the outgoing extension from midpoint toward the end of the shared phoneme and the incoming extension from its start toward midpoint, using the corresponding phoneme timings in each source clip. This gives the join moving frames from both recordings. Use the same rule for a double meeting a triple when they share only one anchor.

Start with a 50 ms overlap centered on the shared anchor. Cap each side at one quarter of the adjacent target anchor interval and at the available motion within the shared phoneme and source clip. Shorten either side independently when there is less room. Across the resulting overlap, blend linearly from 100% outgoing to 100% incoming. If no output timestamp falls strictly inside the fade, average the continuous pose over a one-frame window centered on each output timestamp whose window intersects it. Bake those averages into the 60 fps track. Check `/m/`, `/p/`, and `/b/` on the rendered frames around joins. If a fade weakens a lip closure, test shifting it to one side of the anchor so one source supplies the closure frame at full weight.

For a missed interval, fade from the preceding matched pose to neutral and from neutral into the following matched pose wherever those neighbors exist. Cap each fade at one quarter of the missed interval and hold neutral between fades. For confirmed pauses, place the fades within the low-energy region and hold the adjacent edge pose up to that region if needed. Hold neutral where neither side has a match.

Generate the 60 fps track once per utterance. Use the audio clock to select the animation frame during play, pause, seek, and replay.

## 6. Reuse the studio interface and verify playback

Follow the existing James studio layout, styling, face and mouth close-up controls, head motion control, and **Generate & play** flow. Validate the transcript with TinyAligner before starting Phonon. Show actionable errors beside the input for words missing from the lexicon, numbers that need to be spelled out, and unsupported text. Reuse the validated transcript for alignment. Show **Dictionary** as the face animation engine and report synthesis, alignment, and animation progress. Prepare the complete clip before playback in the first version so full TinyAligner has the utterance and transcript. [Audio2Face studio](../../../audio2face/README.md) · [Full TinyAligner demo](../../../../model-demos/ttp/tinyaligner/README.md)

The first milestone is one development sentence playing end to end on James with diagnostics showing which intervals used a triple, a pair, or neither. Before scaling up, verify overlap endpoints and playback synchronization. For triple-to-pair and pair-to-pair joins, verify that the blend samples recorded frames beyond both nominal spans and that it still works when the available overlap is short or falls between two 60 fps timestamps. Test a quiet pause and a long alignment gap containing audible speech, a missed interval, and single-phoneme inputs with a known match and a known miss. Confirm invalid text is rejected before synthesis. Inspect the actual rendered frames for visible jumps and for closure and release on a sentence ending in `/m/`, `/p/`, or `/b/`.

For the primary comparison, run full v3 on the same test audio, fit the resulting geometry to ARKit, and play both the directly fitted track and the dictionary track on the same James blendshape rig. Use the full v3 geometry playback as a separate reference to assess what the ARKit fit loses. Inspect lip closures and joins, and report coverage by match type, lookup time, time to playback, and browser memory.

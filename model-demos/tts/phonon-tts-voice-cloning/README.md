# Phonon Voice Cloning Lab

A local experiment with the community Phonon speaker encoder. Record your own
voice in the browser, create a reusable voice prompt, and generate English speech
with Phonon 31M. Separate cloned/default voice players let you compare the same
sentence without overwriting either result. This creates conditioning from the
recording; it does not fine-tune Phonon's weights.

## Run

Requires Node 22.18+, uv, Python 3.12, and a browser with microphone access and
AudioWorklet support. Tested on Apple Silicon with Chrome. Everything binds to
loopback; no recording is sent to a cloud service.

```sh
npm ci
npm run setup
npm run dev
```

Open the URL printed by Vite. Development starts at **127.0.0.1:5173**, then tries
the first free port. The private Python model service receives an automatically
allocated port and a temporary internal token from the launcher; it is not a
separately configured public API. Closing the launcher stops both processes.

Setup downloads pinned public assets into ignored `.models/`: the community
encoder implementation/head, Kyutai's pretrained Mimi encoder front, and Phonon
31M's Q8 bundle. The encoder sources and weights are never included in a web build
or committed to this repository. Startup verifies their SHA-256 manifest. The
encoder and speech model remain loaded between calls.

To reuse an existing Phonon bundle without modifying or duplicating it:

```sh
PHONON_BASE_MODEL=/absolute/path/to/phonon-31m-en npm run setup
```

That directory must contain `config.json`, `tokenizer.json`, `model.q8.gguf` and
`default-voice.safetensors`. This optional read-only dependency is recorded in the
local manifest. No other sandbox app or existing inference service is required.
`.env.example` documents optional model/data directories. Set these as environment
variables when running setup and dev; no credentials are needed.

## Use

1. Record 10–15 seconds in a quiet room. The recorder stops at 20 seconds, closes
   the microphone and converts the audio to mono 24 kHz WAV. An existing 5–20
   second recording can also be used.
2. Listen back, name the voice, confirm it is your own and choose **Create my voice**.
3. Enter a new English sentence and choose **Generate speech**.
4. Choose **Generate default voice** to compare it with Phonon's checkpoint preset.
   Both results remain playable. Each records its text; a message identifies
   comparisons made with different sentences. WAV downloads are available.

The reference preparation follows the community experiment: 80 Hz high-pass,
relative silence removal with margins, up to 10 seconds and -18 LUFS normalization.
Quiet, clipped, invalid and oversized recordings are rejected. This silence gate
does not identify speakers or guarantee that a recording contains speech.

Voice profiles, source recordings, embeddings and generated WAVs are stored in
ignored `.data/` with private filesystem permissions. Voices survive app/browser
restarts and port changes. **Delete voice** removes its reference, embedding and
associated generated speech. Default comparison clips are independent of a voice.
Generated players are per-page; saved profiles remain selectable after reload.
Changing the frontend port changes the browser permission/storage origin.

## Experimental personal fitting

The quick encoder pass may change timbre while losing the reference accent. An
optional second pass searches for a closer personal voice prompt while keeping
the encoder, Phonon weights and native synthesis engine fixed.

```sh
npm run setup                 # installs the Python dependencies
npm run setup:refinement      # additional public local models; no recordings uploaded
```

Select a saved voice and open **Refine this voice**. Transcribe its recording
locally, or type exactly what was spoken. Correct the transcript and confirm it
matches the recording. The suggested reading passage is optional; the words in
the speech-generation box are not a recording transcript.

**Create fitted voice** makes a separate profile. Quick fitting runs 20 search
steps (about 1–2 minutes on the tested Mac); longer fitting runs 60 steps. Runtime
depends on reference length and CPU. Progress and cancellation are available;
leaving the page does not overwrite the source voice. If no better-fitting profile
is found, the app reports that and retains the original. A cancelled search does
not publish a partial candidate.

Choose **Voice and pronunciation** for an experimental accent-sensitive pass.
It gives aligned acoustic matching substantially more weight and permits bounded
channel-gain adjustments in addition to mean/timbre offsets. Starting from a fitted
voice preserves that earlier candidate as a separate profile. A new sentence
containing words such as “car”, “castle” and “water” helps listening comparisons.
If speaker similarity drops materially on that comparison sentence, the search
tries smaller adjustments and rejects the candidate if none pass. That sentence
is excluded from the acoustic objective but used for this safeguard; it is not
an independent evaluation. The automated acoustic loss is still not an accent classifier or proof
that the desired accent was preserved. Accurate reference words matter more for
this mode; verify the transcript before fitting.

The search uses paired perturbations in a small PCA space derived from the public
English voice catalogue. A frozen ERes2Net speaker model supplies an identity
comparison; transcript-aligned log-mel DTW, spectral shape and duration terms
encourage acoustic similarity. Fixed seeds and zero sampling temperature make
candidate comparisons repeatable. Synthetic fitting audio is never mistaken for
the original recording.

Saved listening comparisons use temperature **0.3** and seed `4242424242424242`,
the same shared settings as **Generate speech**. The displayed speaker scores
describe these playback clips. The search and its safeguard retain temperature
0 and seed 42; their diagnostic scores are stored separately.

The saved candidate includes separate **Original clone** and **Fitted clone**
players for the same sentence, which was excluded from the acoustic objective.
Pronunciation mode also uses it to reject changes that weaken voice similarity.
Automated speaker
cosines are diagnostic numbers, not percentages, proof of identity, or accent
scores. A score can improve while perceived likeness remains poor. Use new
sentences and listen before choosing a fitted voice. The model's underlying
language/accent limitations still apply. This fits a voice prompt, not a globally
improved audio encoder.

Fitting uses the whole saved recording up to 20 seconds, without the quick
encoder's 10-second truncation. Both profiles keep their own reference copy, so
deleting one does not remove the other. Transcripts, search audio and metrics are
private `.data/` files. User-specific profiles are never committed.

Additional provenance:

- The fitting approach is informed by the public frozen-TTS voice-style search
  in [DavidValin/supertonic3-tts](https://github.com/DavidValin/supertonic3-tts#how-it-works)
  and [saurabhv749/supertonic3-voice-clone](https://github.com/saurabhv749/supertonic3-voice-clone).
  The Phonon search here is implemented independently; their style tensors are
  incompatible with Phonon and are not imported.
- Speaker comparison: [3D-Speaker ERes2Net English VoxCeleb model](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models),
  asset `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx`, Apache 2.0. Model
  authors: 3D-Speaker/ModelScope; ONNX export: sherpa-onnx. Pinned SHA-256 in setup.
- Transcription: [OpenAI Whisper base.en](https://huggingface.co/openai/whisper-base.en),
  revision `911407f4214e0e1d82085af863093ec0b66f9cd6`, Apache 2.0 model-card licence.
- Voice search directions: [Gradium public voice catalogue](https://huggingface.co/gradium/phonon-voices),
  revision `0ee17205f254b5abc43b266c8a0f215565c352f7`, Apache 2.0.

The optional assets are kept in `.models/refinement/`. Setup downloads them once
and records checksums; transcription and fitting then work offline.

## Quality and provenance

This is an experimental clone, not a verified replica. The original author's
evaluation found inconsistent speaker identity despite mostly intelligible
speech. No quality claim is made for a newly recorded voice. Compare the source,
clone and default with your own ears. No model or preset substitution is used
when cloning fails.

- Community implementation/head: [connerohnesorge/phonon-encoder](https://github.com/connerohnesorge/phonon-encoder),
  revision `8cad11e932a6883b23433176522e09fba5ffa974`. The original repository has no
  declared licence; commercial reuse and redistribution need clarification. This
  lab fetches its code only for the explicitly requested local experiment.
- Encoder front: [Kyutai Mimi tokenizer](https://huggingface.co/kyutai/moshiko-pytorch-bf16),
  revision `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`,
  `tokenizer-e351c8d8-checkpoint125.safetensors`, CC BY 4.0. Attribution: Kyutai.
- Speech model: [Gradium Phonon 31M English](https://huggingface.co/gradium/phonon-31m-en),
  revision `92741c5b07d085ff840e26fea8ce43110b87852f`, Apache 2.0.
- The separate Empyrealworks voice encoder trained using 40M outputs is not used.

## Checks

```sh
npm run test
npm run check
npm run build
```

The real-model browser tests need the app running, `npm run setup:refinement`
completed, and an explicitly supplied
5–20 second WAV fixture for Chrome's synthetic microphone. It creates and deletes
only a test profile; it checks microphone capture, encoding, real synthesis,
default comparison, reload persistence, downloads, mobile overflow and origin
rejection. The additional fitting test verifies local transcription, transcript
confirmation, cancellation, creation of a separate fitted voice and comparison
playback. Existing profiles are preserved.

```sh
TEST_BASE_URL=http://127.0.0.1:<printed-port> \
TEST_VOICE_WAV=/absolute/path/to/test-reference.wav npm run test:browser
```

For a built app, run `npm run build` and `npm run preview`. Preview starts both
the built frontend (5173 or the first free port) and persistent inference process.
A standalone Node
deployment would also need the private Python process and shared internal token.
No hosted deployment is configured.

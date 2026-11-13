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

The real-model browser test needs the app running and an explicitly supplied
5–20 second WAV fixture for Chrome's synthetic microphone. It creates and deletes
only a test profile; it checks microphone capture, encoding, real synthesis,
default comparison, reload persistence, downloads, mobile overflow and origin
rejection. Existing profiles are preserved.

```sh
TEST_BASE_URL=http://127.0.0.1:<printed-port> \
TEST_VOICE_WAV=/absolute/path/to/test-reference.wav npm run test:browser
```

For a built app, run `npm run build` and `npm run preview`. Preview starts both
the built frontend (5173 or the first free port) and persistent inference process.
A standalone Node
deployment would also need the private Python process and shared internal token.
No hosted deployment is configured.

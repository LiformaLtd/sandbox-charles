# TinyAligner phoneme timing lab

A standalone SvelteKit demo that generates an English sample with Phonon on page load and runs TinyAligner on the resulting PCM. Edit the sentence and select **Generate & align** to repeat the flow. Play the clip to follow the waveform and highlighted phonemes, or select a word or phoneme to seek to its estimated onset.

## Run locally

Requires Node 22.17 or newer and a browser with WebAssembly SIMD.

```sh
npm ci
npm run assets
npm run dev
```

Open http://127.0.0.1:5186/. There is no required environment configuration. The asset command fetches the pinned TinyAligner model and prepares its English dictionary. The app downloads the Phonon Q8 weights (about 65 MB), tokenizer, and voice from Hugging Face on the first run. Synthesis and alignment run on the device. The text and audio are processed locally.

```sh
npm run check
npm run build
HOST=127.0.0.1 PORT=5186 node build
```

## Timing and input

Use short English sentences with dictionary pronunciations and spell out numbers. The display uses stress-free ARPABET labels and estimated CTC boundaries on a 20 ms grid. The alignment timer includes resampling, feature extraction, inference, and decoding. The first alignment also includes ONNX session setup and loading the model.

The alignment feature extraction and CTC decoder are adapted from [TinyAligner](https://github.com/krmanik/tiny-aligner/tree/1f07ad6d186c70baac4fae26c605cc84660d65f7), whose README identifies its code license as MIT. The English pronunciation data comes from CMUdict. Speech synthesis uses [Gradium Phonon](https://gradium.ai/phonon) with the Easton preset. This app has no dependency on the sibling TTS demo.

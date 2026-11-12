# Pocket TTS Lab

A local SvelteKit 3 sandbox for comparing Kyutai's October 2026 small English models with the original January 2026 release. Generate a sample with one model or run all four sequentially with the same text and voice.

| Parameters including decoder | Upstream model |
| --- | --- |
| 18.2M | `english_18m_2026-10` |
| 6.9M | `english_7m_2026-10` |
| 1.9M | `english_2m_2026-10` |
| 100M (original) | `english_2026-01` |

## Run locally

Requires Node 22.17+ and uv with Python 3.12.

```sh
npm ci
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python -m uvicorn inference:app --host 127.0.0.1 --port 8178
```

In another terminal from this directory:

```sh
npm run dev
```

Open http://127.0.0.1:5178. Both services bind to loopback. If you change the inference port, export `POCKET_TTS_URL` in the SvelteKit terminal before starting it. See `.env.example` for settings.

The Python service lazily downloads model assets and stock voices from Hugging Face and reads them from `~/Source/liforma/ai-workflows/apple-silicon/.pocket-tts-cache` on later runs. Keep that directory out of Git. Set `HF_HUB_CACHE` in the Python terminal to use another cache. The first run needs internet access. Models and voice states stay in memory until the Python service restarts.

Setup time includes any model and voice loading. Generation time measures synthesis after setup. Speed is audio duration divided by generation time. WAV clips can be played and downloaded from each model's card.

Run `npm run check` and `npm run build` to validate the app. To serve the Node build locally, run `HOST=127.0.0.1 PORT=5178 BODY_SIZE_LIMIT=11M node build` with the Python service running.

## Clone a voice

Request access to the [voice-cloning weights](https://huggingface.co/kyutai/pocket-tts), then run `.venv/bin/hf auth login` and restart the inference service if it previously loaded the preset-only weights. Choose **Clone from audio** in the Voice menu and select a reference WAV or M4A up to 10 MB. M4A conversion requires FFmpeg on the inference service’s PATH. The lab processes the first 30 seconds locally and discards the temporary reference file after synthesis. **Compare all** uses the same reference for each model.

## ONNX Streaming

This tab runs the 18.2M REFL checkpoint in a browser worker using ONNX Runtime Web's WASM backend. The first run fetches the pinned public export from Hugging Face and caches the assets in browser storage. Presets and reference audio synthesis run in the browser. For reference audio, WAV and M4A decoding depends on browser codec support.

TTFA measures time to the first playable chunk, including setup and any downloads. Total generation starts after setup. After playback starts, you can download the complete WAV when generation finishes.

Model by [Kyutai](https://huggingface.co/kyutai/pocket-tts), converted by [Maxime Vidal](https://huggingface.co/maxime/pocket-tts-18m-onnx), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

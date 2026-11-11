"""Local Pocket TTS inference service for the sandbox."""

import base64
import io
import json
import logging
import os
from pathlib import Path
from subprocess import DEVNULL, CalledProcessError, TimeoutExpired, run
from tempfile import TemporaryDirectory
from threading import Event, Lock
from time import perf_counter

# The Hub reads its cache setting when Pocket TTS is imported.
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(Path.home() / "Source/liforma/ai-workflows/apple-silicon/.pocket-tts-cache"),
)

from anyio import CancelScope, to_thread
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Form, HTTPException, Response, UploadFile
from pocket_tts import TTSModel
from pydantic import BaseModel, StringConstraints
from scipy.io.wavfile import write
from typing import Annotated, Literal

models = {
    model["id"]
    for model in json.loads((Path(__file__).parent / "src/lib/models.json").read_text())
}
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
lock = Lock()
loaded_models = {}
voice_states = {}
logger = logging.getLogger("uvicorn.error")


class SynthesisRequest(BaseModel):
    model: str
    voice: Literal["alba", "anna", "marius"]
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


@app.post("/synthesize")
def synthesize(request: SynthesisRequest, stream: bool = False):
    return generate(request, stream=stream)


@app.post("/clone")
def clone(model: Annotated[str, Form()], text: Annotated[str, Form(min_length=1, max_length=1000)], reference: UploadFile, stream: bool = False):
    if not text.strip():
        raise HTTPException(400, "Enter text to speak.")
    if reference.size is None or reference.size > 10 * 1024 * 1024:
        raise HTTPException(413, "Choose a WAV or M4A file up to 10 MB.")
    sample = reference.file.read()
    is_wav = sample[:4] == b"RIFF" and sample[8:12] == b"WAVE"
    is_m4a = sample[4:8] == b"ftyp"
    if not (is_wav or is_m4a):
        raise HTTPException(400, "Choose a valid WAV or M4A file.")

    with TemporaryDirectory() as directory:
        wav = Path(directory) / "reference.wav"
        if is_wav:
            wav.write_bytes(sample)
        else:
            source = Path(directory) / "reference.m4a"
            source.write_bytes(sample)
            try:
                run(
                    [
                        "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file",
                        "-f", "mov", "-i", str(source), "-map", "0:a:0", "-t", "30",
                        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(wav),
                    ],
                    stdin=DEVNULL,
                    check=True,
                    timeout=15,
                )
            except FileNotFoundError as exc:
                raise HTTPException(503, "FFmpeg is required to read M4A references.") from exc
            except (CalledProcessError, TimeoutExpired) as exc:
                raise HTTPException(400, "Could not decode the reference M4A.") from exc
        return generate(SynthesisRequest(model=model, text=text, voice="alba"), wav, stream)


def generate(request: SynthesisRequest, reference: Path | None = None, stream: bool = False):
    if request.model not in models:
        raise HTTPException(400, "Unknown Pocket TTS model.")

    try:
        # The upstream model is not thread-safe. Serial runs also avoid CPU contention.
        with lock:
            started = perf_counter()
            cold = request.model not in loaded_models
            if cold:
                loaded_models[request.model] = TTSModel.load_model(language=request.model)
            model = loaded_models[request.model]
            voice_key = (request.model, request.voice)
            if reference is not None:
                if not model.has_voice_cloning:
                    raise HTTPException(
                        503,
                        "Voice cloning needs access to kyutai/pocket-tts on Hugging Face. "
                        "Sign in and restart the inference service after access is granted.",
                    )
                try:
                    voice_state = model.get_state_for_audio_prompt(reference, truncate=True)
                except (ValueError, RuntimeError, OSError) as exc:
                    raise HTTPException(400, "Could not read the reference WAV.") from exc
            else:
                if voice_key not in voice_states:
                    voice_states[voice_key] = model.get_state_for_audio_prompt(request.voice)
                voice_state = voice_states[voice_key]
            prepared = perf_counter()
            if stream:
                return StreamingResponse(
                    stream_audio(model, voice_state, request.text),
                    media_type="application/x-ndjson",
                    headers={
                        "x-setup-ms": f"{(prepared - started) * 1000:.1f}",
                        "x-model-load": "cold" if cold else "warm",
                        "x-sample-rate": str(model.sample_rate),
                    },
                )
            audio = model.generate_audio(voice_state, request.text)
            generated = perf_counter()
        wav = io.BytesIO()
        write(wav, model.sample_rate, audio.numpy())
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Synthesis failed for %s", request.model)
        raise HTTPException(502, "Pocket TTS failed. See the inference terminal for details.") from exc

    return Response(
        wav.getvalue(),
        media_type="audio/wav",
        headers={
            "x-setup-ms": f"{(prepared - started) * 1000:.1f}",
            "x-generation-ms": f"{(generated - prepared) * 1000:.1f}",
            "x-audio-seconds": f"{len(audio) / model.sample_rate:.3f}",
            "x-model-load": "cold" if cold else "warm",
        },
    )


async def stream_audio(model, voice_state, text):
    stop = Event()

    def packets():
        with lock:
            started = perf_counter()
            chunks = model.generate_audio_stream(voice_state, text, stop=stop)
            try:
                for chunk in chunks:
                    pcm = chunk.numpy().astype("<f4", copy=False).tobytes()
                    yield json.dumps({"pcm": base64.b64encode(pcm).decode("ascii")}) + "\n"
                yield json.dumps({"generationMs": (perf_counter() - started) * 1000}) + "\n"
            except Exception:
                logger.exception("Streaming synthesis failed")
                yield json.dumps({"error": "Streaming synthesis failed. See the inference terminal."}) + "\n"
            finally:
                # Let Pocket TTS finish its decoder thread before releasing the model lock.
                stop.set()
                for _ in chunks:
                    pass

    iterator = packets()
    try:
        while (packet := await to_thread.run_sync(next, iterator, None)) is not None:
            yield packet
    finally:
        stop.set()
        with CancelScope(shield=True):
            await to_thread.run_sync(iterator.close)

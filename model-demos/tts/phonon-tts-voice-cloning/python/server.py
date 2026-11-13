"""Loopback-only, persistent CPU encoder and Phonon service for the voice lab."""

import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import secrets
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
import soundfile as sf
import torch
from audio import MAX_BYTES, SAMPLE_RATE, prepare_reference, read_recording
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from safetensors.torch import load_file, save_file

os.umask(0o077)
ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("PHONON_CLONE_MODELS", ROOT / ".models")).resolve()
DATA = Path(os.environ.get("PHONON_CLONE_DATA", ROOT / ".data")).resolve()
TOKEN = os.environ.get("PHONON_CLONE_TOKEN", "")
log = logging.getLogger("phonon-cloning")
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phonon-cloning")
busy = False


def identifier(value):
    try:
        if str(UUID(value)) != value:
            raise ValueError()
    except ValueError:
        raise HTTPException(404, "Voice or recording not found.") from None
    return value


class Engine:
    ready = False
    error = "Starting the local voice engine…"
    refinement = None
    fitter = None
    transcriber = None

    def load(self):
        manifest = json.loads((MODELS / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            with Path(name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"Model checksum mismatch: {Path(name).name}. Run npm run setup."
                )
        spec = importlib.util.spec_from_file_location(
            "community_encoder", MODELS / "community_encoder.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.set_num_threads(4)
        self.encoder = module.MimiEncoder(module.EncCfg()).eval()
        self.encoder.load_state_dict(
            load_file(MODELS / "speaker_encoder.safetensors"), strict=True
        )
        self.base = Path(manifest["base_model"])
        self.reset_synth()
        self.ready, self.error = True, ""

    def reset_synth(self):
        import ptts

        self.synth = ptts.TTS(str(self.base), lang="en", device="cpu", quant="q8")
        self.registered = set()

    def clone(self, raw, name):
        started = perf_counter()
        pcm = read_recording(raw)
        reference = prepare_reference(pcm)
        with torch.inference_mode():
            latents = (
                self.encoder(torch.from_numpy(reference)[None, None])
                .float()
                .contiguous()
            )
        if (
            latents.ndim != 3
            or latents.shape[1] != 512
            or not torch.isfinite(latents).all()
        ):
            raise RuntimeError("The community encoder returned invalid voice latents.")
        voice_id = str(uuid4())
        folder = DATA / "voices" / voice_id
        folder.mkdir(parents=True)
        try:
            # Same format and signature as Gradium's catalogue. No fine-tuning.
            save_file(
                {"speaker_wavs": latents},
                folder / "voice.safetensors",
                metadata={
                    "sig": "ba7a5a45",
                    "epoch": "300",
                    "source": "community-speaker-encoder",
                },
            )
            sf.write(folder / "reference.wav", pcm, SAMPLE_RATE, subtype="PCM_16")
            profile = {
                "id": voice_id,
                "name": name,
                "createdAt": datetime.now(UTC).isoformat(),
                "recordingSeconds": len(pcm) / SAMPLE_RATE,
                "usedSeconds": len(reference) / SAMPLE_RATE,
                "creationMs": (perf_counter() - started) * 1000,
            }
            (folder / "profile.json").write_text(json.dumps(profile))
            return profile
        except Exception:
            shutil.rmtree(folder)
            raise

    def speak(self, text, voice_id):
        started = perf_counter()
        if voice_id != "default":
            folder = voice_folder(voice_id)
            if voice_id not in self.registered:
                self.synth.add_voice(voice_id, str(folder / "voice.safetensors"))
                self.registered.add(voice_id)
        chunks = []
        with self.synth.stream(text, voice=voice_id) as stream:
            for pcm in stream:
                chunks.append(np.array(pcm, dtype=np.float32, copy=True))
        if not chunks:
            raise RuntimeError("Phonon returned no speech.")
        audio = np.concatenate(chunks)
        if not np.isfinite(audio).all():
            raise RuntimeError("Phonon returned invalid audio.")
        result_id = str(uuid4())
        sf.write(
            DATA / "audio" / f"{result_id}.wav", audio, SAMPLE_RATE, subtype="PCM_16"
        )
        result = {
            "id": result_id,
            "voiceId": voice_id,
            "url": f"/api/audio/{result_id}",
            "durationSeconds": len(audio) / SAMPLE_RATE,
            "generationMs": (perf_counter() - started) * 1000,
        }
        (DATA / "audio" / f"{result_id}.json").write_text(json.dumps(result))
        return result

    def verify_refinement_assets(self):
        folder = MODELS / "refinement"
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Local fitting models are not installed. Run npm run setup:refinement."
            )
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["files"].items():
            with Path(name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(
                    "A fitting model failed verification. Run npm run setup:refinement again."
                )
        return folder

    def transcribe(self, voice_id):
        from copy import deepcopy

        from scipy.signal import resample_poly
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            GenerationMixin,
        )

        if self.transcriber is None:
            folder = self.verify_refinement_assets() / "whisper"
            processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                folder, local_files_only=True
            ).eval()
            model.generation_config.max_length = None
            self.transcriber = processor, model
        processor, model = self.transcriber
        audio, _ = sf.read(voice_folder(voice_id) / "reference.wav", dtype="float32")
        inputs = processor(
            resample_poly(audio, 2, 3),
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        config = deepcopy(model.generation_config)
        config.forced_decoder_ids = None
        config.max_new_tokens = 160
        config.max_length = None
        config._from_model_config = False
        # References are <=20 seconds, so normal short-form encoder/decoder
        # generation suffices. Explicit English/no-timestamps prefix avoids the
        # deprecated long-form Whisper helper's forced-token argument handling.
        prefix = torch.tensor(
            [
                [
                    model.config.decoder_start_token_id,
                    processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
                ]
            ]
        )
        with torch.inference_mode():
            ids = GenerationMixin.generate(
                model, **inputs, decoder_input_ids=prefix, generation_config=config
            )
        text = processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        if not text:
            raise ValueError(
                "No transcript could be produced. Please type the words you recorded."
            )
        return {"text": text, "needsReview": True}

    def refine(self, voice_id, transcript, iterations, focus="voice"):
        from refinement import Cancelled, Fitter

        source = voice_folder(voice_id)
        original = json.loads((source / "profile.json").read_text())
        candidate_id, job_id = str(uuid4()), str(uuid4())
        folder = DATA / "voices" / candidate_id
        self.cancel_refinement = threading.Event()
        self.refinement = {
            "id": job_id,
            "voiceId": voice_id,
            "stage": "Preparing local fitting",
            "step": 0,
            "total": iterations,
        }
        try:
            if self.fitter is None:
                self.fitter = Fitter(self.verify_refinement_assets(), self.base)

            def progress(values):
                self.refinement = {**self.refinement, **values}

            metrics = self.fitter.run(
                source,
                transcript,
                folder,
                iterations,
                progress=progress,
                cancel=self.cancel_refinement,
                focus=focus,
            )
            shutil.copyfile(source / "reference.wav", folder / "reference.wav")
            profile = {
                **original,
                "id": candidate_id,
                "name": original["name"][:40]
                + (" · pronunciation fit" if focus == "pronunciation" else " · fitted"),
                "createdAt": datetime.now(UTC).isoformat(),
                "parentId": voice_id,
                "method": "personal-fit",
                "transcript": transcript,
                "fit": metrics,
            }
            (folder / "profile.json").write_text(json.dumps(profile))
            return profile
        except Cancelled:
            return {"cancelled": True}
        finally:
            # Only a complete, successful candidate becomes visible in the voice list.
            if not (folder / "profile.json").is_file():
                shutil.rmtree(folder, ignore_errors=True)
            self.refinement = None


engine = Engine()


async def work(function, *args):
    global busy
    if not engine.ready:
        raise HTTPException(503, engine.error)
    if busy:
        raise HTTPException(
            409, "The voice engine is busy. Wait for the current operation to finish."
        )
    busy = True
    # Shield the actual job: disconnecting must not release the lock while the
    # worker is still modifying profiles or using the shared native model.
    future = asyncio.get_running_loop().run_in_executor(pool, function, *args)
    future.add_done_callback(lambda _: release())
    try:
        return await asyncio.shield(future)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        log.exception("Local voice operation failed")
        raise HTTPException(
            500,
            "The local voice operation failed. Check the service log and try again.",
        ) from error


def release():
    global busy
    busy = False


@asynccontextmanager
async def lifespan(_):
    if not TOKEN:
        raise RuntimeError(
            "Start this app with npm run dev to configure the local service token."
        )
    for path in [DATA / "voices", DATA / "audio"]:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        await asyncio.get_running_loop().run_in_executor(pool, engine.load)
    except Exception:
        log.exception("Could not load voice models")
        engine.error = (
            "Voice models could not be loaded. Run npm run setup, then restart the app."
        )
    yield
    if engine.refinement:
        engine.cancel_refinement.set()
    pool.shutdown(wait=True)


async def authorize(request: Request):
    if not TOKEN or not secrets.compare_digest(
        request.headers.get("x-lab-token", ""), TOKEN
    ):
        raise HTTPException(403, "Use the local voice-cloning app.")


app = FastAPI(
    lifespan=lifespan,
    dependencies=[Depends(authorize)],
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def voice_folder(voice_id):
    folder = DATA / "voices" / identifier(voice_id)
    if not (folder / "profile.json").is_file():
        raise HTTPException(404, "Voice not found.")
    return folder


@app.get("/health")
def health():
    return {
        "ready": engine.ready,
        "error": engine.error,
        "busy": busy,
        "model": "Phonon 31M English",
        "device": "Local CPU · Q8",
        "refinementReady": (MODELS / "refinement" / "manifest.json").is_file(),
        "supportsPronunciation": True,
        "refinement": engine.refinement,
    }


@app.get("/voices")
def voices():
    profiles = [
        json.loads(path.read_text())
        for path in (DATA / "voices").glob("*/profile.json")
    ]
    return sorted(profiles, key=lambda profile: profile["createdAt"], reverse=True)


@app.post("/voices")
async def create_voice(request: Request, name: str = "My voice"):
    if request.headers.get("x-voice-consent") != "own-voice":
        raise HTTPException(
            400, "Confirm that this is your own voice before creating a clone."
        )
    name = name.strip()
    if not 1 <= len(name) <= 60 or any(ord(c) < 32 for c in name):
        raise HTTPException(400, "Give the voice a name of 1–60 characters.")
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, "Recording too large. Record up to 20 seconds.")
    return await work(engine.clone, bytes(raw), name)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    voiceId: str


class RefinementRequest(BaseModel):
    transcript: str = Field(min_length=3, max_length=1000)
    confirmed: bool
    iterations: int = Field(default=40, ge=10, le=80)
    focus: Literal["voice", "pronunciation"] = "voice"


@app.post("/voices/{voice_id}/transcribe")
async def transcribe_voice(voice_id: str):
    voice_folder(voice_id)
    return await work(engine.transcribe, voice_id)


@app.post("/voices/{voice_id}/refine")
async def refine_voice(voice_id: str, body: RefinementRequest):
    voice_folder(voice_id)
    if not body.confirmed:
        raise HTTPException(
            400, "Check the transcript against your recording before fitting."
        )
    if len(body.transcript.strip()) < 3:
        raise HTTPException(400, "Enter the actual words spoken in the recording.")
    return await work(
        engine.refine, voice_id, body.transcript.strip(), body.iterations, body.focus
    )


@app.post("/refinements/{job_id}/cancel")
async def cancel_refinement(job_id: str):
    identifier(job_id)
    if not engine.refinement or engine.refinement["id"] != job_id:
        raise HTTPException(404, "This fitting job is no longer running.")
    engine.cancel_refinement.set()
    return {"cancelling": True}


@app.post("/speech")
async def speech(body: SpeechRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Enter something to say.")
    if body.voiceId != "default":
        voice_folder(body.voiceId)
    return await work(engine.speak, text, body.voiceId)


@app.get("/voices/{voice_id}/{asset}")
def voice_asset(voice_id: str, asset: str):
    folder = voice_folder(voice_id)
    if asset == "reference":
        return FileResponse(
            folder / "reference.wav",
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )
    if asset == "embedding":
        return FileResponse(
            folder / "voice.safetensors",
            filename="cloned-voice.safetensors",
            headers={"Cache-Control": "no-store"},
        )
    if (
        asset in {"original-comparison", "fitted-comparison"}
        and (folder / f"{asset}.wav").is_file()
    ):
        return FileResponse(
            folder / f"{asset}.wav",
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )
    raise HTTPException(404, "Voice asset not found.")


@app.get("/audio/{result_id}")
def audio(result_id: str):
    path = DATA / "audio" / f"{identifier(result_id)}.wav"
    if not path.is_file():
        raise HTTPException(404, "Generated speech not found.")
    return FileResponse(
        path, media_type="audio/wav", headers={"Cache-Control": "no-store"}
    )


def remove_voice(voice_id):
    folder = voice_folder(voice_id)
    for path in (DATA / "audio").glob("*.json"):
        if json.loads(path.read_text())["voiceId"] == voice_id:
            path.with_suffix(".wav").unlink(missing_ok=True)
            path.unlink()
    shutil.rmtree(folder)
    if voice_id in engine.registered:
        engine.reset_synth()
    return {"deleted": True}


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    identifier(voice_id)
    return await work(remove_voice, voice_id)

"""Versioned, atomic batches of directly comparable speech errors."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import soundfile as sf
from audio import SAMPLE_RATE, prepare_reference
from refinement import Scorer, check_cancel
from synthesis import PLAYBACK_SETTINGS


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol(models):
    models = Path(models)
    return {
        "metric": "shared-acoustic-v1",
        "weights": {"speaker": 4, "mel": 0.25, "spectrum": 0.15, "duration": 0.2},
        "parameterPenalty": False,
        "sampling": dict(PLAYBACK_SETTINGS),
        "preparation": "reference-v1-max20-seconds",
        "modelManifestSha256": digest(models / "manifest.json"),
        "speakerSha256": digest(models / "refinement" / "speaker.onnx"),
    }


def comparison_key(recording_sha, transcript, settings):
    payload = {
        "recordingSha256": recording_sha,
        "transcript": transcript.strip(),
        "protocol": settings,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{uuid4()}.tmp")
    try:
        temporary.write_text(json.dumps(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_comparison(
    data, models, source, voice_ids, transcript, render, progress, cancel
):
    started = perf_counter()
    source = Path(source)
    recording_sha = digest(source / "reference.wav")
    # Validate the entire batch before synthesising or publishing anything.
    folders = [Path(data) / "voices" / voice_id for voice_id in voice_ids]
    if any(digest(folder / "reference.wav") != recording_sha for folder in folders):
        raise ValueError("Compare only versions made from the same saved recording.")
    settings = protocol(models)
    key = comparison_key(recording_sha, transcript, settings)
    check_cancel(cancel)
    scorer = Scorer(Path(models) / "refinement" / "speaker.onnx")
    raw, sr = sf.read(source / "reference.wav", dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError("The saved reference has an unexpected sample rate.")
    reference = scorer.features(prepare_reference(raw, max_seconds=20))
    results = []
    maximum = max(10, reference["seconds"] * 2 + 3)
    for index, voice_id in enumerate(voice_ids):
        check_cancel(cancel)
        progress(
            {
                "stage": "Measuring shared errors",
                "step": index,
                "total": len(voice_ids),
                "voiceId": voice_id,
            }
        )
        voice_started = perf_counter()
        pcm = render(transcript, voice_id, max_seconds=maximum, cancel=cancel)
        score = scorer.compare(
            scorer.features(pcm), reference, transcript=True, focus="voice"
        )
        results.append(
            {
                "voiceId": voice_id,
                "error": score["loss"],
                "speakerCosine": score["speakerCosine"],
                "melDistance": score["melDistance"],
                "spectrumDistance": score["spectrumDistance"],
                "durationDistance": score["durationDistance"],
                "elapsedSeconds": perf_counter() - voice_started,
            }
        )
    check_cancel(cancel)
    batch = {
        "id": str(uuid4()),
        "comparisonKey": key,
        "recordingSha256": recording_sha,
        "transcript": transcript,
        "protocol": settings,
        "createdAt": datetime.now(UTC).isoformat(),
        "elapsedSeconds": perf_counter() - started,
        "results": results,
    }
    atomic_json(Path(data) / "evaluations" / f"{batch['id']}.json", batch)
    return batch


def read_batches(data):
    batches = []
    for path in (Path(data) / "evaluations").glob("*.json"):
        try:
            batches.append(json.loads(path.read_text()))
        except FileNotFoundError:
            continue  # A completed deletion may race a read-only poll.
    return batches


def latest_scores(data):
    batches = read_batches(data)
    scores = {}
    for batch in sorted(batches, key=lambda b: b["createdAt"], reverse=True):
        for result in batch["results"]:
            scores.setdefault(
                result["voiceId"],
                {
                    **result,
                    "comparisonKey": batch["comparisonKey"],
                    "transcript": batch["transcript"],
                    "protocol": batch["protocol"],
                    "createdAt": batch["createdAt"],
                },
            )
    return scores


def remove_scores(data, voice_id):
    for path in (Path(data) / "evaluations").glob("*.json"):
        batch = json.loads(path.read_text())
        remaining = [row for row in batch["results"] if row["voiceId"] != voice_id]
        if len(remaining) == len(batch["results"]):
            continue
        if remaining:
            batch["results"] = remaining
            atomic_json(path, batch)
        else:
            path.unlink()

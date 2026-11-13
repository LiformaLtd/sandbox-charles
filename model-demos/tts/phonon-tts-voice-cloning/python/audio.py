"""Bounded recording validation and the community experiment's reference preparation."""

import math
from io import BytesIO

import numpy as np
import pyloudnorm
import soundfile as sf
from scipy import signal

SAMPLE_RATE = 24000
MAX_BYTES = 2_000_000


def read_recording(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Record between 5 and 20 seconds of speech (WAV, up to 2 MB).")
    try:
        with sf.SoundFile(BytesIO(raw)) as recording:
            if (
                recording.format != "WAV"
                or recording.channels != 1
                or recording.samplerate != SAMPLE_RATE
            ):
                raise ValueError("Use a mono, 24 kHz WAV recording.")
            if not 5 * SAMPLE_RATE <= recording.frames <= 20 * SAMPLE_RATE:
                raise ValueError("Record between 5 and 20 seconds of speech.")
            pcm = recording.read(dtype="float32")
    except (sf.LibsndfileError, RuntimeError) as error:
        raise ValueError("The recording is not a readable WAV file.") from error
    if not np.isfinite(pcm).all():
        raise ValueError("The recording contains invalid audio samples.")
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms < 0.002:
        raise ValueError(
            "The recording is too quiet. Move closer to the microphone and try again."
        )
    if np.mean(np.abs(pcm) >= 0.995) > 0.02:
        raise ValueError(
            "The recording is clipping. Lower the microphone level and record again."
        )
    return pcm


def prepare_reference(pcm):
    # Match the author's 80 Hz high-pass, relative silence gate with 150 ms
    # margins, 10 second cap, and -18 LUFS target. This is not speaker detection.
    w = 2 * math.pi * 80 / SAMPLE_RATE
    alpha, cosine = math.sin(w) / math.sqrt(2), math.cos(w)
    b = np.array([1 + cosine, -2 * (1 + cosine), 1 + cosine]) / (2 * (1 + alpha))
    a = np.array([1 + alpha, -2 * cosine, 1 - alpha]) / (1 + alpha)
    filtered = signal.lfilter(b, a, pcm).astype(np.float32)
    frames = filtered[: len(filtered) // 240 * 240].reshape(-1, 240)
    energy = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    threshold = np.sort(energy)[round((len(energy) - 1) * 0.99)] - 30
    keep = np.convolve(energy > threshold, np.ones(31), mode="same") > 0
    condensed = frames[keep].reshape(-1)
    if len(condensed) < SAMPLE_RATE // 2:
        condensed = filtered
    prepared = condensed[: 10 * SAMPLE_RATE].copy()
    prepared -= prepared.mean()
    loudness = pyloudnorm.Meter(SAMPLE_RATE).integrated_loudness(
        prepared.astype(np.float64)
    )
    if math.isfinite(loudness):
        prepared *= 10 ** ((-18 - loudness) / 20)
    if not np.isfinite(prepared).all():
        raise ValueError("Could not prepare the recording. Try a clearer voice sample.")
    return prepared

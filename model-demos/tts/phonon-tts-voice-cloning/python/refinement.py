"""Experimental personal voice fitting through frozen native Phonon inference.

The search updates only a small voice-prompt vector. It uses paired perturbations
(SPSA/NES), speaker similarity, and transcript-aligned acoustic comparisons.
The pretrained encoder and speech weights are never changed.
"""

import math
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import ptts
import soundfile as sf
import torch
import torchaudio
from audio import SAMPLE_RATE, prepare_reference
from safetensors.torch import load_file, save_file
from scipy.signal import resample_poly
from scipy.spatial.distance import cdist
from synthesis import PLAYBACK_SETTINGS

HOLDOUT = "The old stone bridge looks beautiful in the afternoon sunshine."
PRONUNCIATION_HOLDOUT = (
    "I parked the car near the old castle and asked for a glass of water."
)


class Cancelled(Exception):
    pass


def check_cancel(event):
    if event is not None and event.is_set():
        raise Cancelled("Voice fitting was cancelled.")


def comparison_audio(base, original, fitted, text, cancel=None):
    """Render both listening clips exactly as normal Generate speech does."""
    synth = ptts.TTS(str(base), lang="en", device="cpu", quant="q8")
    results = []
    for name, path in (("original", original), ("fitted", fitted)):
        check_cancel(cancel)
        synth.add_voice(name, str(path))
        parts, count = [], 0
        with synth.stream(text, voice=name, **PLAYBACK_SETTINGS) as stream:
            for chunk in stream:
                check_cancel(cancel)
                pcm = np.array(chunk, dtype=np.float32, copy=True)
                parts.append(pcm)
                count += len(pcm)
                if count > 30 * SAMPLE_RATE:
                    raise ValueError("A comparison clip produced overlong speech.")
        if not parts:
            raise ValueError("A comparison clip produced no speech.")
        audio = np.concatenate(parts)
        if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 0.001:
            raise ValueError("A comparison clip produced invalid or silent speech.")
        results.append(audio)
    return results


class Scorer:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.model = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

    def features(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if not np.isfinite(audio).all() or len(audio) < SAMPLE_RATE // 2:
            raise ValueError("Not enough valid speech to compare voices.")
        wave = torch.from_numpy(resample_poly(audio, 2, 3).copy())[None]
        # Match the exported model's normalize_samples=1 and global-mean metadata.
        feats = torchaudio.compliance.kaldi.fbank(
            wave,
            sample_frequency=16000,
            num_mel_bins=80,
            dither=0,
            snip_edges=False,
            high_freq=-400,
        ).numpy()
        normal = feats - feats.mean(0, keepdims=True)
        embedding = self.model.run(None, {"x": normal[None].astype(np.float32)})[
            0
        ].reshape(-1)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
        # Log-mel statistics use energetic frames; global loudness is removed.
        energy = np.mean(feats, axis=1)
        active = feats[energy > np.quantile(energy, 0.25)]
        if len(active) < 10:
            active = feats
        centered = active - np.mean(active)
        return {
            "embedding": embedding,
            "mel": feats[::4] - np.mean(feats),
            "spectrum": centered.mean(0),
            "seconds": len(audio) / SAMPLE_RATE,
        }

    def compare(self, features, reference, transcript=True, focus="voice"):
        cosine = float(np.clip(features["embedding"] @ reference["embedding"], -1, 1))
        spectrum = (
            float(np.mean((features["spectrum"] - reference["spectrum"]) ** 2)) / 25
        )
        mel, duration = 0.0, 0.0
        if transcript:
            import librosa

            costs = cdist(features["mel"], reference["mel"], metric="sqeuclidean") / 80
            accumulated, path = librosa.sequence.dtw(C=costs)
            mel = float(accumulated[-1, -1] / len(path)) / 25
            duration = (math.log(features["seconds"] / reference["seconds"])) ** 2
        acoustic_weight = 3.0 if focus == "pronunciation" else 0.25
        spectrum_weight = 0.4 if focus == "pronunciation" else 0.15
        duration_weight = 0.35 if focus == "pronunciation" else 0.2
        return {
            "loss": 4 * (1 - cosine)
            + acoustic_weight * mel
            + spectrum_weight * spectrum
            + duration_weight * duration,
            "speakerCosine": cosine,
            "melDistance": mel,
            "durationDistance": duration,
        }


def make_basis(paths, output, rank=16):
    """PCA over public voice prompts gives the search sensible, bounded directions."""
    rows = []
    for path in paths:
        state = load_file(path)
        if "speaker_wavs" not in state:
            continue
        voice = state["speaker_wavs"].float()
        voice = torch.nn.functional.interpolate(
            voice, size=125, mode="linear", align_corners=False
        )
        rows.append(voice[0].T.reshape(-1).numpy())
    if len(rows) < 8:
        raise ValueError(
            "At least eight public catalogue voices are needed for fitting."
        )
    rows = np.stack(rows)
    rows -= rows.mean(0)
    values, vectors = np.linalg.eigh(rows @ rows.T)
    order = np.argsort(values)[::-1][: min(rank, len(rows) - 1)]
    basis = (vectors[:, order].T @ rows) / np.sqrt(len(rows) - 1)
    np.savez_compressed(output, basis=basis.reshape(-1, 125, 512).astype(np.float32))


class Fitter:
    def __init__(self, models, base):
        self.models, self.base = Path(models), Path(base)
        self.scorer = Scorer(self.models / "speaker.onnx")
        self.basis = torch.from_numpy(np.load(self.models / "basis.npz")["basis"])

    def run(
        self,
        folder,
        transcript,
        output,
        iterations=40,
        progress=None,
        cancel=None,
        focus="voice",
    ):
        started = perf_counter()
        if focus not in {"voice", "pronunciation"}:
            raise ValueError("Choose voice similarity or pronunciation fitting.")
        holdout_text = PRONUNCIATION_HOLDOUT if focus == "pronunciation" else HOLDOUT
        Path(output).mkdir(parents=True, exist_ok=True)
        raw, sr = sf.read(Path(folder) / "reference.wav", dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError("The saved reference has an unexpected sample rate.")
        reference = self.scorer.features(prepare_reference(raw, max_seconds=20))
        original = load_file(Path(folder) / "voice.safetensors")["speaker_wavs"].float()
        # Search mean/timbre offsets. Broadcasting the learned channel offset to
        # all frames avoids changing the reference's time order/content.
        basis = self.basis.mean(1)
        rank = len(basis)
        dimension = rank * (2 if focus == "pronunciation" else 1)
        gain_directions = basis / torch.sqrt(
            (basis**2).mean(1, keepdim=True)
        ).clamp_min(0.05)
        synth = ptts.TTS(
            str(self.base), lang="en", device="cpu", quant="q8", temperature=0
        )
        temporary = Path(output) / "search.safetensors"
        counter = 0

        def prompt(coefficients):
            values = torch.from_numpy(coefficients.astype(np.float32))
            offset = values[:rank] @ basis
            if focus == "pronunciation":
                # Channel gains preserve frame order while adjusting more than
                # the first pass's mean/timbre offset.
                gain = (
                    1 + 0.15 * (values[rank:] @ gain_directions) / math.sqrt(rank)
                ).clamp(0.65, 1.35)
            else:
                gain = torch.ones(512)
            return (original * gain[None, :, None] + offset[None, :, None]).contiguous()

        def generate(coefficients, text):
            nonlocal counter, synth
            check_cancel(cancel)
            # New identifiers avoid any reuse of primed voice caches. Periodic
            # model renewal bounds the native voice registry during a long fit.
            if counter and counter % 24 == 0:
                synth = ptts.TTS(
                    str(self.base), lang="en", device="cpu", quant="q8", temperature=0
                )
            name = f"fit-{counter}"
            counter += 1
            save_file(
                {"speaker_wavs": prompt(coefficients)},
                temporary,
                metadata={"sig": "ba7a5a45", "epoch": "300"},
            )
            synth.add_voice(name, str(temporary))
            parts, count = [], 0
            maximum = int(max(10, reference["seconds"] * 2 + 3) * SAMPLE_RATE)
            with synth.stream(text, voice=name, temperature=0, seed=42) as stream:
                for chunk in stream:
                    check_cancel(cancel)
                    pcm = np.array(chunk, dtype=np.float32, copy=True)
                    parts.append(pcm)
                    count += len(pcm)
                    if count > maximum:
                        raise ValueError(
                            "A fitting candidate produced overlong speech."
                        )
            if not parts:
                raise ValueError("A fitting candidate produced no speech.")
            audio = np.concatenate(parts)
            if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 0.001:
                raise ValueError(
                    "A fitting candidate produced invalid or silent speech."
                )
            return audio

        def evaluate(coefficients, text=transcript, aligned=True):
            audio = generate(coefficients, text)
            metrics = self.scorer.compare(
                self.scorer.features(audio), reference, aligned, focus
            )
            metrics["loss"] += 0.015 * float(np.mean(coefficients**2))
            return metrics, audio

        center = np.zeros(dimension, dtype=np.float64)
        before, before_audio = evaluate(center)
        held_before, _ = evaluate(center, holdout_text, False)
        best, best_metrics = center.copy(), before.copy()
        rng = np.random.default_rng(42)
        momentum, variance = np.zeros(dimension), np.zeros(dimension)
        Path(output).mkdir(parents=True, exist_ok=True)
        sf.write(
            Path(output) / "before.wav", before_audio, SAMPLE_RATE, subtype="PCM_16"
        )
        for step in range(iterations):
            check_cancel(cancel)
            sigma = 0.12 / (1 + step / 20) ** 0.1
            gradient = np.zeros(dimension)
            for _ in range(3):
                direction = rng.choice([-1.0, 1.0], size=dimension)
                positive = np.clip(center + sigma * direction, -0.8, 0.8)
                negative = np.clip(center - sigma * direction, -0.8, 0.8)
                plus, _ = evaluate(positive)
                minus, _ = evaluate(negative)
                gradient += (plus["loss"] - minus["loss"]) / (2 * sigma) * direction / 3
                for coefficients, metrics in [(positive, plus), (negative, minus)]:
                    if metrics["loss"] < best_metrics["loss"]:
                        best, best_metrics = coefficients.copy(), metrics.copy()
            gradient = np.clip(gradient, -2, 2)
            momentum = 0.9 * momentum + 0.1 * gradient
            variance = 0.999 * variance + 0.001 * gradient**2
            update = (momentum / (1 - 0.9 ** (step + 1))) / (
                np.sqrt(variance / (1 - 0.999 ** (step + 1))) + 1e-8
            )
            center = np.clip(center - 0.06 * update, -0.8, 0.8)
            current, _ = evaluate(center)
            if current["loss"] < best_metrics["loss"]:
                best, best_metrics = center.copy(), current.copy()
            if progress:
                progress(
                    {
                        "stage": "Fitting your voice",
                        "step": step + 1,
                        "total": iterations,
                        "before": before["speakerCosine"],
                        "best": best_metrics["speakerCosine"],
                        "elapsedSeconds": perf_counter() - started,
                    }
                )
        after, after_audio = evaluate(best)
        held_after, _ = evaluate(best, holdout_text, False)
        retained_scale = 1.0
        # A stronger acoustic objective can overfit the recorded words. Backtrack
        # toward the starting profile until a smaller, still useful change also
        # preserves similarity on the comparison sentence.
        if (
            focus == "pronunciation"
            and held_after["speakerCosine"] < held_before["speakerCosine"] - 0.02
        ):
            for scale in (0.75, 0.5, 0.25, 0.125, 0.0625):
                trial = best * scale
                trial_metrics, trial_audio = evaluate(trial)
                if trial_metrics["loss"] >= before["loss"] - 0.001:
                    continue
                trial_held, _ = evaluate(trial, holdout_text, False)
                if trial_held["speakerCosine"] >= held_before["speakerCosine"] - 0.02:
                    best, retained_scale = trial, scale
                    after, after_audio = trial_metrics, trial_audio
                    held_after = trial_held
                    break
            else:
                raise ValueError(
                    "The pronunciation fit weakened voice similarity on new text, so it was not saved. Your previous voice is unchanged."
                )
        sf.write(Path(output) / "after.wav", after_audio, SAMPLE_RATE, subtype="PCM_16")
        temporary.unlink(missing_ok=True)
        if after["loss"] >= before["loss"] - 0.001:
            raise ValueError(
                "The search did not find a better-fitting profile. Your original voice is unchanged."
            )
        save_file(
            {"speaker_wavs": prompt(best)},
            Path(output) / "voice.safetensors",
            metadata={
                "sig": "ba7a5a45",
                "epoch": "300",
                "source": "experimental-personal-voice-fit",
            },
        )
        playback_before, playback_after = comparison_audio(
            self.base,
            Path(folder) / "voice.safetensors",
            Path(output) / "voice.safetensors",
            holdout_text,
            cancel,
        )
        sf.write(
            Path(output) / "original-comparison.wav",
            playback_before,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        sf.write(
            Path(output) / "fitted-comparison.wav",
            playback_after,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        playback_before_metrics = self.scorer.compare(
            self.scorer.features(playback_before), reference, False, focus
        )
        playback_after_metrics = self.scorer.compare(
            self.scorer.features(playback_after), reference, False, focus
        )
        return {
            "before": before,
            "after": after,
            "searchHeldOutBefore": held_before,
            "searchHeldOutAfter": held_after,
            "heldOutBefore": playback_before_metrics,
            "heldOutAfter": playback_after_metrics,
            "comparisonSettings": dict(PLAYBACK_SETTINGS),
            "elapsedSeconds": perf_counter() - started,
            "iterations": iterations,
            "focus": focus,
            "retainedScale": retained_scale,
            "heldOutText": holdout_text,
            "generalized": playback_after_metrics["speakerCosine"]
            > playback_before_metrics["speakerCosine"],
        }

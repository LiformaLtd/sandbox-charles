"""One regularized solve in the full centred catalogue space, without TTS search."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from audio import SAMPLE_RATE, prepare_reference
from refinement import HOLDOUT, Scorer, check_cancel, comparison_audio
from safetensors.torch import save_file
from scipy.linalg import helmert
from synthesis import PLAYBACK_SETTINGS

RIDGE = 0.05
BANK_TEXTS = (
    "The morning air feels fresh and there is plenty of time to enjoy a quiet walk.",
    "Please leave a message after the signal and I will call you back later today.",
)


def linear_error(features, target, weights, ridge=RIDGE):
    """The exact objective minimized by solve_weights, including the ridge term."""
    features, target, weights = (
        np.asarray(value, dtype=np.float64) for value in (features, target, weights)
    )
    residual = float(np.sum((weights @ features - target) ** 2))
    penalty = float(ridge * np.sum((weights - 1 / len(weights)) ** 2))
    return residual + penalty, residual, penalty


def solve_weights(features, target, ridge=RIDGE):
    """Fit all N-1 contrasts; sum(weights)=1, with ridge allowing negative weights.

    An orthonormal Helmert basis spans exactly the same full centred catalogue
    space as all nonzero PCA components, without truncating it to 16 directions.
    """
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2 or target.shape != features.shape[1:]:
        raise ValueError("Invalid catalogue or reference speaker features.")
    if (
        not np.isfinite(features).all()
        or not np.isfinite(target).all()
        or not np.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("Invalid linear fit values or regularization.")
    contrasts = helmert(len(features), full=False).T
    centered = features - features.mean(0)
    design = centered.T @ contrasts
    coefficients = np.linalg.solve(
        design.T @ design + ridge * np.eye(design.shape[1]),
        design.T @ (target - features.mean(0)),
    )
    weights = np.ones(len(features)) / len(features) + contrasts @ coefficients
    return weights, int(np.linalg.matrix_rank(design))


def load_bank(models, base):
    folder = Path(models) / "linear"
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
        for file, checksum in manifest["files"].items():
            with (folder / file).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                    raise ValueError(
                        "Linear fitting data failed verification. Run npm run setup:linear again."
                    )
        base_manifest = json.loads((Path(models).parent / "manifest.json").read_text())
        with (Path(models) / "speaker.onnx").open("rb") as stream:
            speaker_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            manifest["baseManifestSha256"]
            != hashlib.sha256(
                json.dumps(base_manifest, sort_keys=True).encode()
            ).hexdigest()
            or Path(manifest["baseModel"]) != Path(base)
            or manifest.get("speakerSha256") != speaker_sha
        ):
            raise ValueError(
                "The linear catalogue was built for a different model. Run npm run setup:linear again."
            )
        with np.load(folder / "catalogue.npz", allow_pickle=False) as bank:
            features, voices, names = bank["features"], bank["voices"], bank["names"]
        if (
            features.shape != (37, 192)
            or voices.shape != (37, 512, 125)
            or names.shape != (37,)
            or not np.isfinite(features).all()
            or not np.isfinite(voices).all()
        ):
            raise ValueError(
                "Invalid linear catalogue. Run npm run setup:linear again."
            )
        return features, voices, names
    except FileNotFoundError as error:
        raise ValueError(
            "Fast linear fitting data is not installed. Run npm run setup:linear."
        ) from error


class LinearFitter:
    def __init__(self, models, base):
        self.base = Path(base)
        self.scorer = Scorer(Path(models) / "speaker.onnx")
        self.features, self.voices, self.names = load_bank(models, base)

    def run(self, folder, output, progress=None, cancel=None):
        started = perf_counter()
        check_cancel(cancel)
        if progress:
            progress({"stage": "Measuring your recorded voice", "step": 0, "total": 3})
        raw, sr = sf.read(Path(folder) / "reference.wav", dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError("The saved reference has an unexpected sample rate.")
        prepared = prepare_reference(raw, max_seconds=20)
        reference = self.scorer.features(prepared)
        solve_started = perf_counter()
        weights, rank = solve_weights(self.features, reference["embedding"])
        final_error, residual_error, ridge_penalty = linear_error(
            self.features, reference["embedding"], weights
        )
        voice = np.tensordot(weights, self.voices, axes=(0, 0)).astype(np.float32)[None]
        solve_ms = (perf_counter() - solve_started) * 1000
        if not np.isfinite(voice).all():
            raise ValueError("The linear solve produced an invalid voice.")
        check_cancel(cancel)
        Path(output).mkdir(parents=True, exist_ok=True)
        save_file(
            {"speaker_wavs": torch.from_numpy(voice)},
            Path(output) / "voice.safetensors",
            metadata={
                "sig": "ba7a5a45",
                "epoch": "300",
                "source": "experimental-linear-voice-fit",
            },
        )
        if progress:
            progress({"stage": "Generating comparison clips", "step": 1, "total": 3})
        before, after = comparison_audio(
            self.base,
            Path(folder) / "voice.safetensors",
            Path(output) / "voice.safetensors",
            HOLDOUT,
            cancel,
        )
        check_cancel(cancel)
        before_metrics = self.scorer.compare(
            self.scorer.features(before), reference, False
        )
        after_metrics = self.scorer.compare(
            self.scorer.features(after), reference, False
        )
        sf.write(
            Path(output) / "original-comparison.wav",
            before,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        sf.write(
            Path(output) / "fitted-comparison.wav", after, SAMPLE_RATE, subtype="PCM_16"
        )
        if progress:
            progress({"stage": "Saving linear fit", "step": 3, "total": 3})
        return {
            "algorithm": "linear",
            "components": 36,
            "effectiveRank": rank,
            "catalogueVoices": 37,
            "referenceSeconds": len(prepared) / SAMPLE_RATE,
            "ridge": RIDGE,
            "finalError": final_error,
            "residualError": residual_error,
            "ridgePenalty": ridge_penalty,
            "solveMs": solve_ms,
            "elapsedSeconds": perf_counter() - started,
            "weights": [
                {"name": str(name), "weight": float(weight)}
                for name, weight in zip(self.names, weights)
            ],
            "before": before_metrics,
            "after": after_metrics,
            "heldOutBefore": before_metrics,
            "heldOutAfter": after_metrics,
            "heldOutText": HOLDOUT,
            "comparisonSettings": dict(PLAYBACK_SETTINGS),
            "generalized": after_metrics["speakerCosine"]
            > before_metrics["speakerCosine"],
        }

"""Fit full native MLX Audio2Face v3 geometry to the existing James ARKit rig."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

# Nine established James lip landmarks, ordered from the left corner to the right.
LIP_PAIRS = np.array([[9013, 10613], [941, 10433], [8859, 10434],
                      [2963, 3726], [1439, 1475], [5980, 7505],
                      [6093, 18810], [263, 18809], [17362, 13358]])


def prepare_fit(rig_dir):
    with np.load(rig_dir / "bs_skin_James.npz") as rig, np.load(
        rig_dir / "model_data_James.npz"
    ) as model:
        config = json.loads((rig_dir / "bs_skin_config_James.json").read_text())
        channels = [name.decode() for name in rig["poseNames"][1:]]
        active = np.flatnonzero(config["blendshape_params"]["bsSolveActivePoses"])
        mask = rig["frontalMask"]
        skin = model["neutral_skin"]
        mouth = (np.abs(skin[:, 0]) < 3) & (skin[:, 1] > 166.1) & (skin[:, 1] < 169) & (skin[:, 2] > 5)
        weights = np.repeat(np.where(mouth[mask], 4., 1.).astype(np.float32), 3)
        basis = np.stack([rig[channels[i]][mask].ravel() for i in active]).T.copy()
        neutral = rig["neutral"]
        offset = (skin - neutral)[mask].ravel()
        upper, lower = LIP_PAIRS.T
        lip_baseline = skin[upper, 1] - skin[lower, 1]
        lip_rig = {
            "neutral": (neutral[upper, 1] - neutral[lower, 1]).tolist(),
            "deltas": [(rig[name][upper, 1] - rig[name][lower, 1]).tolist() for name in channels],
        }
    if len(channels) != 52:
        raise ValueError("Expected the 52 James ARKit channels")
    weighted_basis = basis * weights[:, None]
    gram = basis.T @ weighted_basis + np.eye(len(active), dtype=np.float32)
    inverse = np.linalg.inv(gram)
    step = np.float32(1 / np.linalg.eigvalsh(gram)[-1])

    def fit(geometry):
        skin_delta = geometry[:, :72006].reshape(-1, 24002, 3)
        rhs = (skin_delta[:, mask].reshape(len(geometry), -1) + offset) @ weighted_basis
        current = np.clip(rhs @ inverse, 0, 1)
        extrapolated = current
        momentum = 1.
        for _ in range(95):
            updated = np.clip(extrapolated - step * (extrapolated @ gram - rhs), 0, 1)
            next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
            extrapolated = updated + np.float32((momentum - 1) / next_momentum) * (updated - current)
            current, momentum = updated, next_momentum
        coefficients = np.zeros((len(geometry), 52), dtype=np.float32)
        coefficients[:, active] = current
        return coefficients

    def geometry_gaps(geometry):
        xyz = geometry[:, :72006].reshape(-1, 24002, 3)
        return lip_baseline + xyz[:, upper, 1] - xyz[:, lower, 1]

    return channels, fit, lip_rig, geometry_gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--geometry-reference", action="store_true",
                        help="Save full geometry lip landmarks for every clip")
    args = parser.parse_args()
    sys.path.insert(0, str(args.runner.resolve()))
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from app.audio2face_v3 import animate, load_backend

    rows = json.loads((args.directory / "clips.json").read_text())
    channels, fit, lip_rig, geometry_gaps = prepare_fit(args.rig)
    (args.directory / "channels.json").write_text(json.dumps(channels))
    (args.directory / "lip-rig.json").write_text(json.dumps(lip_rig))
    infer = None
    for index, clip in enumerate(rows, 1):
        path = args.directory / (clip["id"] + ".arkit")
        audio = np.fromfile(args.directory / (clip["id"] + ".pcm"), dtype="<f4")
        frame_count = (len(audio) * 60 + 15999) // 16000
        gap_path = path.with_suffix(".geometry-gaps")
        reference = args.geometry_reference or clip["split"] == "test"
        have_weights = path.exists() and path.stat().st_size == frame_count * 52 * 4
        have_gaps = gap_path.exists() and gap_path.stat().st_size == frame_count * len(LIP_PAIRS) * 4
        if have_weights and (not reference or have_gaps):
            continue
        if infer is None:
            infer = load_backend(args.model, "mlx")
        blocks = []
        gaps = []
        cursor = 0
        for indices, geometry in animate(audio, infer, identity="James", seed=0):
            expected = np.arange(cursor, cursor + len(indices)) * 16000 // 60
            if not np.array_equal(indices, expected):
                raise ValueError(f"{clip['id']}: animation timestamps are off the 60 fps grid")
            if not have_weights:
                blocks.append(fit(geometry))
            if reference and not have_gaps:
                gaps.append(geometry_gaps(geometry))
            cursor += len(indices)
        if cursor != frame_count:
            raise ValueError(f"{clip['id']}: incomplete ARKit track")
        for destination, arrays, columns in [(path, blocks, 52), (gap_path, gaps, len(LIP_PAIRS))]:
            if not arrays:
                continue
            values = np.concatenate(arrays)
            if values.shape != (frame_count, columns) or not np.isfinite(values).all():
                raise ValueError(f"{clip['id']}: invalid animation data")
            temporary = destination.with_suffix(destination.suffix + ".partial")
            values.astype("<f4").tofile(temporary)
            temporary.replace(destination)
        print(f"{index}/{len(rows)} {clip['id']} {frame_count} frames", flush=True)


if __name__ == "__main__":
    main()

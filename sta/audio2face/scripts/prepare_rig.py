"""Extract matching indexed meshes from the official James Maya ASCII scene."""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def maya_nodes(scene):
    headers = list(re.finditer(r'^createNode (\w+) -n "([^"]+)"(?: -p "([^"]+)")?;', scene, re.M))
    return {
        match[2]: (match[1], match[3], scene[match.end():headers[i+1].start()] if i+1 < len(headers) else scene[match.end():])
        for i, match in enumerate(headers)
    }


def chunks(body, attr, dtype, width):
    pattern = rf'setAttr "\.{attr}\[(\d+)(?::(\d+))?\]"\s+([^;]+);'
    parts = []
    count = 0
    for start, end, values in re.findall(pattern, body):
        assert int(start) == count, f"Unexpected {attr} index {start}"
        count = int(end or start) + 1
        parts.append(np.fromstring(values, sep=" ", dtype=dtype).reshape(-1, width))
    result = np.concatenate(parts)
    assert len(result) == count
    return result


def attribute(body, name, kind, default):
    pattern = rf'setAttr "\.{name}" -type "{kind}"\s+([^;]+);'
    match = re.search(pattern, body)
    return np.fromstring(match[1], sep=" ") if match else np.asarray(default)


def mesh(nodes, name):
    _, parent, body = nodes[name]
    vertices = chunks(body, "vt", np.float32, 3)
    edges = chunks(body, "ed", np.int32, 3)[:, :2]
    triangles = []
    for count, values in re.findall(r'^\s*f (\d+) ([^\n;]+)', body, re.M):
        face = np.fromstring(values, sep=" ", dtype=np.int32)
        assert len(face) == int(count)
        ids = np.where(face < 0, -face - 1, face)
        polygon = edges[ids, (face < 0).astype(np.int32)]
        assert np.array_equal(edges[ids, (face >= 0).astype(np.int32)], np.roll(polygon, -1))
        for i in range(1, len(polygon) - 1):
            triangles.extend((polygon[0], polygon[i], polygon[i+1]))
    if not triangles:
        raise ValueError(f"No faces found for {name}")
    return vertices, np.asarray(triangles, dtype="<u4"), parent


def offset_transform(nodes, parent, positions):
    homogeneous = np.column_stack((positions, np.ones(len(positions))))
    while parent:
        _, parent, body = nodes[parent]
        opm = attribute(body, "opm", "matrix", np.eye(4).ravel()).reshape(4, 4)
        homogeneous = homogeneous @ opm
    return homogeneous[:, :3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("model_data", type=Path)
    parser.add_argument("--out", type=Path, default=Path(".local"))
    args = parser.parse_args()
    nodes = maya_nodes(args.scene.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    specs = {
        "skin": "c_headWatertight_hiShape",
        "tongue": "c_tongue_hiShape",
        "upperTeeth": "c_topDenture_hiShape",
        "lowerTeeth": "c_bottomDenture_hiShape",
        "rightEye": "r_choroid_hiShape",
        "leftEye": "l_choroid_hiShape",
    }
    model = np.load(args.model_data)
    eye_close = model["eye_close_pose_delta"]
    if eye_close.shape != model["neutral_skin"].shape or not np.isfinite(eye_close).all():
        raise ValueError("The eye-close pose must match the neutral skin")
    manifest = {"jaw": model["neutral_jaw"].tolist(), "meshes": {}}
    with (args.out / "rig.bin").open("wb") as binary:
        for key, name in specs.items():
            points, indices, parent = mesh(nodes, name)
            posed = offset_transform(nodes, parent, points)
            meta = {}
            if key in ("skin", "tongue"):
                rest = model[f"neutral_{key}"]
                if len(points) != len(rest):
                    raise ValueError(f"{key} vertex count differs from model data")
                errors = [float(np.linalg.norm(x-rest, axis=1).mean()) for x in (points, posed)]
                if min(errors) > 0.75:
                    raise ValueError(f"{key} vertices do not match the model order: {errors}")
                print(f"{key} {len(points)} vertices; mean neutral pose difference {min(errors):.4f}")
                posed = rest
            elif key.endswith("Eye"):
                group = ("r" if key == "rightEye" else "l") + "_eye_grp_hi"
                _, _, attrs = nodes[group]
                meta["pivot"] = attribute(attrs, "rp", "double3", (0, 0, 0)).tolist()
                meta["rotation"] = attribute(attrs, "r", "double3", (0, 0, 0)).tolist()
            meta.update(positionOffset=binary.tell(), vertexCount=len(posed))
            binary.write(np.asarray(posed, dtype="<f4").tobytes())
            meta.update(indexOffset=binary.tell(), indexCount=len(indices))
            binary.write(indices.tobytes())
            manifest["meshes"][key] = meta
        manifest["eyeCloseOffset"] = binary.tell()
        binary.write(np.asarray(eye_close, dtype="<f4").tobytes())
    (args.out / "rig.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

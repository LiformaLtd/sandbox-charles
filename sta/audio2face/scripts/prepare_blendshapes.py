"""Pack NVIDIA's James ARKit targets for the browser demo."""
import argparse
import json
from pathlib import Path
import re
import struct

import numpy as np

from prepare_rig import chunks, maya_nodes, offset_transform


def maya_deltas(nodes, node, mesh, names):
    body = nodes[node][2]
    aliases = dict(re.findall(r'"([^"]+)"\s*,\s*"weight\[(\d+)\]"', body))
    _, parent, original = nodes[mesh]
    neutral = chunks(original, 'vt', np.float32, 3)
    world = offset_transform(nodes, parent, neutral)
    shapes = {}
    for name in names:
        target = rf'\.it\[0\]\.itg\[{aliases[name]}\]\.iti\[6000\]'
        points = re.search(rf'"{target}\.ipt" -type "pointArray" (\d+)\s+([^;]+);', body)
        if not points:
            continue
        components = re.search(rf'"{target}\.ict" -type "componentList" ([^;]+);', body)
        ranges = re.findall(r'"vtx\[(\d+)(?::(\d+))?\]"', components[1])
        indices = np.concatenate([np.arange(int(lo), int(hi or lo) + 1) for lo, hi in ranges])
        delta = np.fromstring(points[2], sep=' ', dtype=np.float32).reshape(-1, 4)[:, :3]
        if len(delta) != len(indices) or len(delta) != int(points[1]):
            raise ValueError(f'Maya target {name} has inconsistent components')
        result = np.zeros_like(neutral)
        result[indices] = offset_transform(nodes, parent, neutral[indices] + delta) - world[indices]
        shapes[name] = result
    return world, shapes


def jaw_delta(neutral, delta, anchors):
    target = neutral + delta
    center = neutral.mean(axis=0)
    moved_center = target.mean(axis=0)
    u, _, v = np.linalg.svd((neutral - center).T @ (target - moved_center))
    u[:, -1] *= np.linalg.det(u @ v)
    rotation = u @ v
    moved = (anchors - center) @ rotation + moved_center
    return moved - anchors


def pack_group(file, shapes, names, active):
    records = []
    for index in (*active, 52):
        name = names[index] if index < 52 else 'neutralOffset'
        if name not in shapes:
            continue
        shape = shapes[name]
        keep = np.flatnonzero(np.max(np.abs(shape), axis=1) >= .002)
        if not len(keep):
            continue
        values = shape[keep]
        scale = float(np.max(np.abs(values))) / 32767
        packed = np.empty((len(keep), 4), dtype='<i2')
        packed[:, 0] = keep
        packed[:, 1:] = np.rint(values / scale).astype('<i2')
        records.append((struct.pack('<IIf', index, len(keep), scale), packed))
    file.write(struct.pack('<I', len(records)))
    for header, packed in records:
        file.write(header)
        file.write(packed.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('skin', 'config', 'model_data', 'scene', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    skin = np.load(args.skin)
    model = np.load(args.model_data)
    config = json.loads(args.config.read_text())['blendshape_params']
    names = [name.decode() for name in skin['poseNames'][1:]]
    active = np.flatnonzero(config['bsSolveActivePoses']).tolist()
    active_names = [names[i] for i in active]
    nodes = maya_nodes(args.scene.read_text())
    neutral_tongue, tongue = maya_deltas(nodes, 'c_tongue_hi_bs', 'c_tongue_hiShapeOrig', active_names)
    neutral_teeth, teeth = maya_deltas(nodes, 'c_bottomDenture_hi_bs', 'c_bottomDenture_hiShapeOrig', active_names)
    if neutral_tongue.shape != model['neutral_tongue'].shape:
        raise ValueError('Tongue vertex count differs from the James geometry')
    jaw = {name: jaw_delta(neutral_teeth, delta, model['neutral_jaw']) for name, delta in teeth.items()}
    skin_shapes = {name: skin[name] for name in active_names}
    skin_shapes['neutralOffset'] = skin['neutral'] - model['neutral_skin']
    with args.output.open('wb') as file:
        file.write(b'JBS1')
        for group in (skin_shapes, tongue, jaw):
            pack_group(file, group, names, active)


if __name__ == '__main__':
    main()

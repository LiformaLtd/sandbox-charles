"""Frozen backbone features, with clip-safe history across original block boundaries."""
import json, sys, os, importlib.util
import numpy as np
import torch
from common import APP, OUT, WORK, BASELINE, base_module, sha, freeze
from decoder import encoded


def backbone():
    import onnx
    from onnx import numpy_helper
    base = base_module()
    values = {v.name: numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer}
    active = np.load(WORK / 'arkit-student/rig-metrics.npz')['active']
    student = base.Student(96, values['student.mean'], values['student.scale'], active)
    state = {k: torch.from_numpy(values['student.' + k].copy()) for k in student.state_dict()}
    student.load_state_dict(state); student.eval(); student.requires_grad_(False)
    return student, active


def cohorts():
    return [('primary-train', APP / '.local/compact-coverage-v2/coverage', 'train'),
            ('primary-dev', APP / '.local/compact-coverage-v2/coverage', 'dev'),
            ('replay-train', WORK / 'arkit-student-v3', 'train'),
            ('extra-train', WORK / 'arkit-student-v3/extra', 'train'),
            ('replay-dev', WORK / 'arkit-student-v3', 'dev')]


def main():
    torch.set_num_threads(4)
    os.environ['A2F_ABLATION_OUTPUT'] = str(OUT / 'waveform-cache')
    sys.path.insert(0, str(APP / 'scripts/research/compact_ablation'))
    import train as cached
    student, active = backbone()
    locks = {}
    for name, root, split in cohorts():
        target = OUT / 'features' / (name + '.npz'); target.parent.mkdir(exist_ok=True)
        digest = cached.input_fingerprint(str(root), split)
        locks[name] = digest
        if target.exists():
            with np.load(target) as z:
                if str(z['input_sha256']) != digest: raise ValueError('Changed source')
            continue
        mel, weights, gaps, valid = cached.dataset(root, split, active, 16000)
        with torch.no_grad():
            feature = torch.cat([encoded(student, torch.from_numpy(v))[:, 15:45] for v in np.array_split(mel, max(1, (len(mel) + 63) // 64))]).numpy()
        rows = [r for r in json.loads((root / 'manifest.json').read_text()) if r['split'] == split]
        clips, cursor = [], 0
        for row in rows:
            samples = __import__('pathlib').Path(row['audio']).stat().st_size // 4
            n = len(range(-8000, samples - 4000, 8000)); mask = valid[cursor:cursor + n]
            x = feature[cursor:cursor + n][mask]
            y, g = weights[cursor:cursor + n][mask], gaps[cursor:cursor + n][mask]
            with np.load(root / 'labels' / (row['stem'] + '.npz')) as z:
                if not np.allclose(y, z['weights'][:, active]): raise ValueError('Track stitching mismatch')
            voice = row.get('engine', 'replay') + ':' + row.get('voice', 'Easton')
            clips.append((x, y, g, voice)); cursor += n
        if cursor != len(feature): raise ValueError('Incomplete feature coverage')
        lengths = [len(v[0]) for v in clips]
        np.savez(target, features=np.concatenate([v[0] for v in clips]), weights=np.concatenate([v[1] for v in clips]),
                 gaps=np.concatenate([v[2] for v in clips]), lengths=lengths, voices=np.array([v[3] for v in clips]), input_sha256=digest)
        print(name, len(clips), sum(lengths), 'valid frames', flush=True)
    freeze(OUT / 'data-lock.json', dict(inputs=locks, baseline_sha256=sha(BASELINE), source_sha256=sha(__file__)))
    training = [np.load(OUT / 'features' / (name + '.npz'))['features'] for name, _, split in cohorts() if split == 'train']
    values = np.concatenate(training)
    center, spread = values.mean(0), np.maximum(values.std(0), .05)
    norm = OUT / 'normalization.npz'
    if not norm.exists(): np.savez(norm, center=center, spread=spread)
    freeze(OUT / 'normalization-lock.json', dict(sha256=sha(norm), source='training frames only, including historical replay'))


def dataset(name):
    with np.load(OUT / 'features' / (name + '.npz')) as z:
        features, targets, gaps = z['features'], z['weights'], z['gaps']
        lengths, voices = z['lengths'], z['voices']
    with np.load(OUT / 'normalization.npz') as z: center, spread = z['center'], z['spread']
    student, _ = backbone()
    with torch.no_grad():
        baseline = (torch.from_numpy(features) @ student.output.weight[..., 0].T + student.output.bias) * student.scale + student.mean
        baseline = baseline.clamp(0, 1).numpy()
    values = {k: [] for k in ['contexts', 'baseline', 'weights', 'gaps', 'valid', 'voice']}
    cursor = 0
    for n, voice in zip(lengths, voices):
        normalized = (features[cursor:cursor + n] - center) / spread
        padded = np.pad(normalized, ((7, 0), (0, 0)))
        contexts = np.lib.stride_tricks.sliding_window_view(padded, 8, axis=0).transpose(0, 2, 1)
        for start in range(0, n, 30):
            end = min(start + 30, n); count = end - start
            for key, arr in [('contexts', contexts), ('baseline', baseline[cursor:cursor + n]), ('weights', targets[cursor:cursor + n]), ('gaps', gaps[cursor:cursor + n])]:
                block = np.zeros((30, *arr.shape[1:]), np.float32); block[:count] = arr[start:end]; values[key].append(block)
            values['valid'].append(np.arange(30) < count); values['voice'].append(str(voice))
        cursor += n
    return {k: torch.from_numpy(np.asarray(v)) if k != 'voice' else np.array(v) for k, v in values.items()}


if __name__ == '__main__': main()

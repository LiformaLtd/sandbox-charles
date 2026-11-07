"""Freeze a headroom-safe, teacher-labelled gain follow-up without editing v4."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np

APP = Path(__file__).resolve().parents[3]
PILOT = APP / '.local/compact-multivoice'
OUT = Path(os.environ.get('A2F_GAIN_OUTPUT', APP / '.local/compact-gain-v2')).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gain_for_peak(peak, requested):
    if requested <= 1 or peak == 0:
        return requested
    return max(1., min(requested, .98 / peak))


def frozen_write(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f'Frozen artifact changed: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = APP / '.local/compact-v4-validation/v4-candidate.onnx'
    if sha(baseline) != 'b4a275acf78b982fb6ec6fc55333c5cf25613e72b6d9b3d80e95d6e74ac6261d':
        raise ValueError('Unexpected v4 initialization')
    for name in ['model.py', 'label.py', 'mlx_features.py']:
        frozen_write(OUT / 'sources' / name, (PILOT / 'sources' / name).read_bytes())
    for name in ['instrumentation-parity.json', 'phone-classes.json']:
        frozen_write(OUT / name, (PILOT / name).read_bytes())
    sources = [Path(__file__), APP / 'scripts/research/multivoice/train.py',
               APP / 'scripts/research/multivoice/label.py',
               APP / 'scripts/research/multivoice/evaluate.py',
               APP / 'docs/plans/compact-gain-robustness.md']
    protocol = dict(initialization=str(baseline), initialization_sha256=sha(baseline),
                    pilot_manifest_sha256=sha(PILOT / 'manifest.json'), seed=37,
                    epochs=24, steps_per_epoch=31, clean_replay=True,
                    conditions=['original', 'half', 'boost'], extra_test_condition='quarter',
                    headroom_policy='amplification <= min(2, .98/peak), but never attenuate original high peaks',
                    source_sha256={str(p.relative_to(APP)): sha(p) for p in sources})
    frozen_write(OUT / 'protocol.json', json.dumps(protocol, indent=2).encode())
    rows = []
    (OUT / 'labels').mkdir(exist_ok=True)
    for source in json.loads((PILOT / 'manifest.json').read_text()):
        if sha(source['audio']) != source['sha256_audio']:
            raise ValueError('Source audio changed')
        wave = np.fromfile(source['audio'], '<f4')
        peak = float(np.abs(wave).max())
        conditions = [('original', 1.), ('half', .5), ('boost', 2.)]
        if source['split'] == 'test':
            conditions.append(('quarter', .25))
        for condition, requested in conditions:
            gain = gain_for_peak(peak, requested)
            row = dict(source, stem=source['stem'] + '-' + condition,
                       original_stem=source['stem'], condition=condition,
                       requested_gain=requested, actual_gain=gain, original_peak=peak)
            audio = OUT / 'audio' / (row['stem'] + '.f32')
            raw = (wave * np.float32(gain)).astype('<f4').tobytes()
            frozen_write(audio, raw)
            row.update(audio=str(audio), sha256_audio=sha(audio), peak=float(np.abs(wave * gain).max()))
            rows.append(row)
    frozen_write(OUT / 'manifest.json', json.dumps(rows, indent=2).encode())
    for arm in ['control', 'augmented']:
        root = OUT / arm
        root.mkdir(exist_ok=True)
        for name in ['sources', 'labels', 'phone-classes.json']:
            target = root / name
            if not target.exists():
                target.symlink_to(OUT / name, target_is_directory=name != 'phone-classes.json')
        selected = [r for r in rows if r['split'] != 'train' or arm == 'augmented' or r['condition'] == 'original']
        frozen_write(root / 'manifest.json', json.dumps(selected, indent=2).encode())
    print(json.dumps(dict(recordings=len(rows), new_labels=sum(not (OUT / 'labels' / (r['stem'] + '.npz')).exists() for r in rows),
                          boost_no_headroom=sum(r['condition'] == 'boost' and r['actual_gain'] == 1 for r in rows)), indent=2))


if __name__ == '__main__':
    main()

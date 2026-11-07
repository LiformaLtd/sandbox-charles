"""Build the same labelled gain conditions and matched arm manifests."""
import hashlib
import json
import numpy as np
from prepare import OUT, GAIN, freeze, sha


def main():
    base = OUT / 'base'; target = OUT / 'variants'
    target.mkdir(exist_ok=True)
    for name in ['sources', 'phone-classes.json', 'instrumentation-parity.json', 'assets']:
        path = target / name
        if not path.exists():
            path.symlink_to(base / name, target_is_directory=name in ['sources', 'assets'])
    labels = target / 'labels'; labels.mkdir(exist_ok=True)
    base_rows = json.loads((base / 'manifest.json').read_text())
    if len(base_rows) != 416 or not all((base / 'labels' / (r['stem'] + '.npz')).exists() for r in base_rows):
        raise ValueError('Complete aligned base corpus and teacher labels required')
    rows = []
    for source in base_rows:
        wave = np.fromfile(source['audio'], '<f4')
        if sha(source['audio']) != source['sha256_audio']:
            raise ValueError('Changed base recording')
        peak = float(np.abs(wave).max())
        conditions = [('original', 1.), ('half', .5), ('boost', 2.)]
        if source['split'] == 'test':
            conditions.append(('quarter', .25))
        for condition, requested in conditions:
            gain = requested if requested <= 1 or peak == 0 else max(1., min(requested, .98 / peak))
            row = dict(source, stem=source['stem'] + '-' + condition, condition=condition,
                       requested_gain=requested, actual_gain=gain, original_stem=source['stem'])
            audio = target / 'audio' / (row['stem'] + '.f32')
            raw = (wave * np.float32(gain)).astype('<f4').tobytes()
            freeze(audio, raw)
            row.update(audio=str(audio), sha256_audio=sha(audio), peak=float(np.max(np.abs(wave * np.float32(gain)))))
            label = labels / (row['stem'] + '.npz')
            if row['sha256_audio'] == source['sha256_audio'] and not label.exists():
                label.symlink_to(base / 'labels' / (source['stem'] + '.npz'))
            rows.append(row)
    freeze(target / 'manifest.json', rows)
    previous = json.loads((GAIN / 'manifest.json').read_text())
    shared_dev = [r for r in previous if r['split'] == 'dev'] + [r for r in rows if r['split'] == 'dev']
    for arm in ['control', 'coverage']:
        root = OUT / arm; root.mkdir(exist_ok=True)
        for name in ['sources', 'phone-classes.json']:
            if not (root / name).exists():
                (root / name).symlink_to(base / name, target_is_directory=name == 'sources')
        (root / 'labels').mkdir(exist_ok=True)
        selected = [r for r in (previous if arm == 'control' else rows) if r['split'] == 'train']
        selected += shared_dev + [r for r in rows if r['split'] == 'test']
        for row in selected:
            source_root = target if row['stem'].startswith('coverage-') else GAIN
            label = root / 'labels' / (row['stem'] + '.npz')
            if not label.is_symlink():
                label.symlink_to(source_root / 'labels' / label.name)
        freeze(root / 'manifest.json', selected)
    freeze(OUT / 'variant-source-lock.json', dict(script_sha256=sha(__file__),
           base_manifest_sha256=sha(base / 'manifest.json'), old_gain_manifest_sha256=sha(GAIN / 'manifest.json')))
    print(json.dumps(dict(variant_recordings=len(rows), shared_dev_recordings=len(shared_dev),
                          control_train=768, coverage_train=768), indent=2))


if __name__ == '__main__':
    main()

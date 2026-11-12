"""Fresh confirmation levels, using the unchanged teacher-labelled gain policy."""
import json
import numpy as np
from prepare import OUT, freeze, sha


def main():
    base = OUT / 'confirmation/base'; root = OUT / 'confirmation/variants'
    root.mkdir(exist_ok=True)
    for name in ['sources', 'assets', 'phone-classes.json', 'instrumentation-parity.json']:
        if not (root / name).exists():
            (root / name).symlink_to(base / name, target_is_directory=name in ['sources', 'assets'])
    (root / 'labels').mkdir(exist_ok=True)
    original = json.loads((base / 'manifest.json').read_text())
    if len(original) != 96:
        raise ValueError('Expected complete 96-recording confirmation')
    rows = []
    for source in original:
        if sha(source['audio']) != source['sha256_audio'] or not (base / 'labels' / (source['stem'] + '.npz')).exists():
            raise ValueError('Changed waveform or missing base label')
        wave = np.fromfile(source['audio'], '<f4'); peak = float(np.abs(wave).max())
        for condition, requested in [('original', 1.), ('half', .5), ('boost', 2.), ('quarter', .25)]:
            gain = requested if requested <= 1 or peak == 0 else max(1., min(requested, .98 / peak))
            row = dict(source, stem=source['stem'] + '-' + condition, condition=condition,
                       requested_gain=requested, actual_gain=gain)
            audio = root / 'audio' / (row['stem'] + '.f32')
            freeze(audio, (wave * np.float32(gain)).astype('<f4').tobytes())
            row.update(audio=str(audio), sha256_audio=sha(audio))
            label = root / 'labels' / (row['stem'] + '.npz')
            if row['sha256_audio'] == source['sha256_audio'] and not label.exists():
                label.symlink_to(base / 'labels' / (source['stem'] + '.npz'))
            rows.append(row)
    freeze(root / 'manifest.json', rows)
    freeze(OUT / 'confirmation-variant-lock.json', dict(source_sha256=sha(__file__), manifest_sha256=sha(root / 'manifest.json')))
    print('Prepared', len(rows), 'fresh confirmation variants; changed levels require new teacher labels.')


if __name__ == '__main__':
    main()

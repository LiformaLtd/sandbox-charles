"""Merge completed per-engine views without writing through frozen sources."""
import json
import numpy as np
from prepare import OUT as STUDY, freeze, sha
OUT = STUDY / 'confirmation'


def main():
    base = OUT / 'base'
    planned = json.loads((base / 'planned.json').read_text())
    collected, recipes = {}, []
    for engine in ['phonon', 'supertonic']:
        root = OUT / ('base-' + engine)
        rows = json.loads((root / 'manifest.json').read_text())
        if len(rows) != 48:
            raise ValueError('Incomplete aligned engine corpus')
        recipes.append(json.loads((root / 'label-recipe.json').read_text()))
        for row in rows:
            if sha(row['audio']) != row['sha256_audio']:
                raise ValueError('Changed synthesis')
            label = root / 'labels' / (row['stem'] + '.npz')
            with np.load(label) as z:
                frames = (row['samples'] * 60 + 15999) // 16000
                if z['weights'].shape != (frames, 52) or z['gaps'].shape != (frames, 9) or str(z['audio_sha256']) != row['sha256_audio']:
                    raise ValueError('Incomplete or mismatched teacher labels')
            target = base / 'labels' / label.name
            target.parent.mkdir(exist_ok=True)
            if not target.is_symlink():
                target.symlink_to(label)
            collected[row['stem']] = row
    if recipes[0] != recipes[1] or len(collected) != len(planned):
        raise ValueError('Mixed teacher recipe or missing recordings')
    freeze(base / 'label-recipe.json', recipes[0])
    freeze(base / 'instrumentation-parity.json', (OUT / 'base-phonon/instrumentation-parity.json').read_bytes())
    freeze(base / 'manifest.json', [collected[r['stem']] for r in planned])
    freeze(OUT / 'merge-source-lock.json', dict(script_sha256=sha(__file__), manifest_sha256=sha(base / 'manifest.json')))
    print('Merged', len(planned), 'complete aligned/labeled recordings under one teacher recipe.')


if __name__ == '__main__':
    main()

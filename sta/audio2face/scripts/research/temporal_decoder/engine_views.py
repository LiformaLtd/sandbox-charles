"""Prepare owned per-engine alignment/label outputs while synthesis overlaps."""
import json
from prepare import OUT as STUDY, freeze, sha
OUT = STUDY / 'confirmation'


def main():
    base = OUT / 'base'
    rows = json.loads((base / 'planned.json').read_text())
    for engine in ['phonon', 'supertonic']:
        root = OUT / ('base-' + engine); root.mkdir(exist_ok=True)
        for path in (base / 'sources').rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                freeze(root / 'sources' / path.relative_to(base / 'sources'), path.read_bytes())
        for name in ['assets', 'synthesis']:
            if not (root / name).exists():
                (root / name).symlink_to(base / name, target_is_directory=True)
        for name in ['assets.json', 'phone-classes.json']:
            freeze(root / name, (base / name).read_bytes())
        freeze(root / 'planned.json', [r for r in rows if r['engine'] == engine])
    freeze(OUT / 'engine-view-source-lock.json', dict(script_sha256=sha(__file__), planned_sha256=sha(base / 'planned.json')))
    print('Per-engine outputs own their mutable alignment and teacher source files.')


if __name__ == '__main__':
    main()

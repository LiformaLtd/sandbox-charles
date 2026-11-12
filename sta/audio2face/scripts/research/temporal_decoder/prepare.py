"""Freeze matched articulation-state inputs and new confirmation speech."""
from pathlib import Path
import hashlib
import json
import os
import re

APP = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('A2F_TEMPORAL_OUTPUT', APP / '.local/compact-temporal-v1')).resolve()
COVERAGE = APP / '.local/compact-coverage-v2'
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    raw = json.dumps(value, indent=2).encode() if isinstance(value, (dict, list)) else value
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Frozen artifact changed: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)


def normalize(text):
    return ' '.join(re.findall('[a-z]+', text.lower()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'sources/model.py', (COVERAGE / 'coverage/sources/model.py').read_bytes())
    source = COVERAGE / 'coverage'
    for arm in ['control', 'temporal']:
        root = OUT / arm; root.mkdir(exist_ok=True)
        for name in ['sources', 'labels', 'phone-classes.json']:
            if not (root / name).exists():
                (root / name).symlink_to(source / name, target_is_directory=name != 'phone-classes.json')
        freeze(root / 'manifest.json', (source / 'manifest.json').read_bytes())
    baseline = source / 'runs/control-replay-s37/model.onnx'
    if sha(baseline) != '5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd':
        raise ValueError('Wrong published coverage-r2 baseline')
    confirm = OUT / 'confirmation/base'; confirm.mkdir(parents=True, exist_ok=True)
    base = COVERAGE / 'base'
    for path in (base / 'sources').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            freeze(confirm / 'sources' / path.relative_to(base / 'sources'), path.read_bytes())
    if not (confirm / 'assets').exists():
        (confirm / 'assets').symlink_to(base / 'assets', target_is_directory=True)
    for name in ['assets.json', 'phone-classes.json']:
        freeze(confirm / name, (base / name).read_bytes())
    config = json.loads((base / 'assets.json').read_text())
    dictionary = json.loads((base / 'assets/aligner/lexicon.json').read_text())['lexicon']
    prompts = json.loads((HERE / 'confirmation-prompts.json').read_text())
    prior = set()
    prior.update(normalize(r['text']) for r in json.loads((APP / '.local/compact-state-v1/confirmation/base/planned.json').read_text()))
    prior.update(normalize(r['text']) for r in json.loads((APP / '.local/compact-capacity-v1/confirmation/base/planned.json').read_text()))
    for root in [APP / '.local/compact-multivoice', APP / '.local/compact-v4-confirmation', COVERAGE / 'base']:
        prior.update(normalize(r['text']) for r in json.loads((root / 'manifest.json').read_text()))
    for name in ['arkit-student', 'arkit-student-v2', 'arkit-student-v3', 'arkit-student-v3/extra']:
        prior.update(normalize(r['text']) for r in json.loads((WORK / name / 'manifest.json').read_text()))
    planned = []
    for i, (kind, text) in enumerate(prompts):
        if normalize(text) in prior:
            raise ValueError('Historical/duplicate confirmation text: ' + text)
        prior.add(normalize(text))
        unknown = [w for w in re.findall("[A-Za-z]+(?:'[A-Za-z]+)*", text) if w.upper() not in dictionary]
        if unknown:
            raise ValueError('Unknown confirmation words: ' + str(unknown))
        for voice in [v for v in config['voices'] if v['split'] == 'test']:
            planned.append(dict(stem=f'temporal-test-{voice["engine"]}-{voice["name"].lower()}-{i:03}',
                split='test', engine=voice['engine'], voice=voice['name'], gender=voice['gender'], locale=voice['locale'],
                text=text, kind=kind, temperature=.7, supertonic_steps=8, supertonic_speed=1.05))
    freeze(confirm / 'planned.json', planned)
    files = [HERE / n for n in ['train.py', 'decoder.py', 'data.py', 'common.py', 'confirmation-prompts.json', 'prepare.py']]
    files.append(APP / 'docs/plans/compact-temporal-decoder.md')
    protocol = dict(initialization=str(baseline), initialization_sha256=sha(baseline), widths=[96], arms=['control', 'temporal'], seed=37,
                    epochs=24, steps_per_epoch=31, manifest_sha256=sha(source / 'manifest.json'),
                    confirmation_planned_sha256=sha(confirm / 'planned.json'),
                    source_sha256={str(p.relative_to(APP)): sha(p) for p in files})
    freeze(OUT / 'protocol.json', protocol)
    for path in files:
        freeze(OUT / 'sources/frozen' / path.relative_to(APP), path.read_bytes())
    print('Frozen temporal/history-free comparison and', len(planned), 'fresh confirmation recordings.')


if __name__ == '__main__':
    main()

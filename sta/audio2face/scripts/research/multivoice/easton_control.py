"""Post-pilot diagnostic: matched updates/texts with Easton-only training audio.

Generate an explicit frozen copy instead of changing the completed pilot recipe.
This is a diagnostic after the first test inspection, not a fresh confirmation.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
APP = HERE.parents[2]
OUT = Path(os.environ.get('A2F_MULTIVOICE_OUTPUT', APP / '.local/compact-multivoice')).resolve()


def main():
    rows = [r for r in json.loads((OUT / 'manifest.json').read_text()) if r['split'] == 'train']
    voices = sorted({r['engine'] + ':' + r['voice'] for r in rows})
    easton = voices.index('phonon:Easton')
    path = HERE / 'train.py'
    frozen = OUT / 'sources/train-first-pilot.py'
    original = frozen.read_text() if frozen.exists() else path.read_text()
    source = original
    replacements = [
        ('APP = Path(__file__).resolve().parents[3]', f'APP = Path({str(APP)!r})'),
        (("run = OUT / 'runs' / f'{args.arm}-s{args.seed}'" if "f'{args.arm}-s{args.seed}'" in source
          else "run = OUT / 'runs' / f'{args.arm}{suffix}-s{args.seed}'"),
         "run = OUT / 'runs' / f'easton-control-s{args.seed}'"),
        ("groups = [torch.nonzero(train['voice'] == v, as_tuple=True)[0] for v in train['voice'].unique()]",
         f"groups = [torch.nonzero(train['voice'] == {easton}, as_tuple=True)[0]] * 8"),
        ("phone_weight=.1, feature_weight=.1, selection=", "diagnostic='matched Easton-only sampling; all other new recordings excluded from optimizer updates', phone_weight=.1, feature_weight=.1, selection="),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f'Training source changed: {before}')
        source = source.replace(before, after)
    target = OUT / 'sources/train_easton_control.py'
    if target.exists() and target.read_text() != source:
        raise ValueError('Diagnostic recipe changed; use a new study directory')
    target.write_text(source)
    (OUT / 'sources/train-first-pilot.py').write_text(original)
    (OUT / 'easton-control-recipe.json').write_text(json.dumps(dict(
        original_training_sha256=hashlib.sha256(original.encode()).hexdigest(),
        generated_training_sha256=hashlib.sha256(source.encode()).hexdigest(),
        training_voice='phonon:Easton', validation='same four pilot voices',
        classification='post-pilot diagnostic; not a new independent test'), indent=2))
    subprocess.run([sys.executable, str(target), '--arm', 'control', *sys.argv[1:]], check=True)


if __name__ == '__main__':
    main()

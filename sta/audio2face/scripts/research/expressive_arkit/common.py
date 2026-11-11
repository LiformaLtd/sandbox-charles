"""Explicit, read-only source paths for the expressive teacher experiment."""
import hashlib
import json
import os
from pathlib import Path

APP = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('A2F_EXPRESSIVE_OUTPUT', APP / '.local/expressive-arkit-v2')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()
TEACHER = Path(os.environ['A2F_TEACHER_ROOT']).resolve()
ASSETS = Path(os.environ.get('A2F_ASSET_ROOT', WORK / 'v3-model')).resolve()
SOURCE_APP = Path(os.environ['A2F_SOURCE_APP']).resolve()
BASELINE = SOURCE_APP / '.local/compact-feature-v1/control/model.onnx'
EXPECTED_BASELINE = 'ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da'
EMOTIONS = json.loads((ASSETS / 'network_info.json').read_text())['params']['emotions']


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def freeze(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f'Frozen artifact changed: {path}')
    if not path.exists():
        path.write_bytes(raw)

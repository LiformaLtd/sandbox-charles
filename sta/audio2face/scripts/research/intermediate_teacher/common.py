"""Read-only input contracts and frozen outputs for LIFORMA-104."""
from pathlib import Path
import hashlib, json, os, sys
APP = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('A2F_INTERMEDIATE_OUTPUT', APP / '.local/intermediate-teacher-v1')).resolve()
DATA = Path(os.environ['A2F_INTERMEDIATE_INPUT']).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()
SCALE = DATA / '.local/compact-scale-v1'
BASELINE = DATA / '.local/compact-feature-v1/control/model.onnx'
EXPECTED = 'ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da'
os.environ['A2F_SCALE_OUTPUT'] = str(SCALE)
os.environ['MLX_ENABLE_TF32'] = '0'
sys.path.insert(0, str(APP / 'scripts/research/data_scale'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, allow_nan=False).encode()
    if path.exists() and path.read_bytes() != raw: raise ValueError('Frozen artifact changed: '+str(path))
    if not path.exists(): path.write_bytes(raw)
def dataset(names):
    from features import Dataset
    return Dataset(names)
def lock():
    value = json.loads((OUT/'protocol.json').read_text())
    if sha(BASELINE) != EXPECTED: raise ValueError('Current baseline changed')
    for path, digest in value['source_sha256'].items():
        if sha(APP/path) != digest: raise ValueError('Frozen source changed: '+path)
    return value

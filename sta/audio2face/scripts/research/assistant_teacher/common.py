"""Paths shared by the isolated assistant-teacher pilot."""
from pathlib import Path
import hashlib, json, os, sys
APP = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = APP / '.local/compact-assistant-v1'
SCALE = APP / '.local/compact-scale-v1'
WORK = Path(os.environ['A2F_RESEARCH_WORK'])
TEACHER = Path(os.environ['A2F_TEACHER_ROOT'])
BASELINE = APP / '.local/compact-feature-v1/control/model.onnx'
os.environ['A2F_SCALE_OUTPUT'] = str(SCALE)
os.environ['MLX_ENABLE_TF32'] = '0'
sys.path.insert(0, str(APP / 'scripts/research/data_scale'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def freeze(path, value):
    raw = json.dumps(value, indent=2).encode() if isinstance(value, (dict,list)) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Frozen artifact changed: ' + str(path))
    if not path.exists(): path.write_bytes(raw)

from pathlib import Path
import os, json, hashlib, importlib.util, sys
if os.environ.get('A2F_ONNX_SITE_PACKAGES'): sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
APP = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('A2F_TEMPORAL_OUTPUT', APP / '.local/compact-temporal-v1')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()
COVERAGE = APP / '.local/compact-coverage-v2'
BASELINE = COVERAGE / 'coverage/runs/control-replay-s37/model.onnx'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    raw = json.dumps(value, indent=2).encode() if isinstance(value, (list, dict)) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Frozen artifact changed: ' + str(path))
    if not path.exists(): path.write_bytes(raw)


def base_module():
    spec = importlib.util.spec_from_file_location('temporal_frozen_base', OUT / 'sources/model.py')
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    return base

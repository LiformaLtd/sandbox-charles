"""Read-only expanded-corpus inputs; isolated LIFORMA-105 outputs."""
from pathlib import Path
import hashlib,json,os,sys
APP=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('A2F_FRONTEND_OUTPUT',APP/'.local/trainable-frontend-v2')).resolve()
DATA=Path(os.environ['A2F_FRONTEND_INPUT']).resolve();WORK=Path(os.environ['A2F_RESEARCH_WORK']).resolve()
SCALE=DATA/'.local/compact-scale-v1';BASELINE=DATA/'.local/compact-feature-v1/control/model.onnx'
POWER=Path(os.environ.get('A2F_FRONTEND_POWER',APP/'.local/trainable-frontend-v1/power')).resolve()
EXPECTED='ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da'
os.environ['A2F_SCALE_OUTPUT']=str(SCALE);os.environ['MLX_ENABLE_TF32']='0'
sys.path.insert(0,str(APP/'scripts/research/data_scale'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(value,indent=2,allow_nan=False).encode()
 if path.exists() and path.read_bytes()!=raw:raise ValueError('Frozen artifact changed: '+str(path))
 if not path.exists():path.write_bytes(raw)
def lock():
 p=json.loads((OUT/'protocol.json').read_text())
 if sha(BASELINE)!=EXPECTED:raise ValueError('Baseline changed')
 for name,digest in p['source_sha256'].items():
  if sha(APP/name)!=digest:raise ValueError('Frozen source changed: '+name)
 return p

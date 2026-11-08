from pathlib import Path
import os,sys,json,hashlib
APP=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=APP/'.local/compact-protected-v1';SCALE=APP/'.local/compact-scale-v1';WORK=Path(os.environ['A2F_RESEARCH_WORK']);BASELINE=APP/'.local/compact-syncnet-v1/control/model.onnx'
FEATURE=APP/'.local/compact-feature-v1';TARGETED=APP/'.local/compact-targeted-v1';TEACHER=Path(os.environ['A2F_TEACHER_ROOT']);ASSETS=WORK/'v3-model'
EXPECTED='63ddc0c64851edc445eba1364e8b160a8a680c4d3b66638fc57a17b782cfe867'
os.environ['MLX_ENABLE_TF32']='0';os.environ['A2F_SCALE_OUTPUT']=str(SCALE)
sys.path.insert(0,str(APP/'scripts/research/data_scale'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(p,v):
 raw=json.dumps(v,indent=2).encode() if isinstance(v,(dict,list)) else v
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=raw:raise ValueError('Changed frozen artifact '+str(p))
 if not p.exists():p.write_bytes(raw)

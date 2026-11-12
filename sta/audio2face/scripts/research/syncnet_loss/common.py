from pathlib import Path
import os,sys,json,hashlib
APP=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=APP/'.local/compact-syncnet-v1';SCALE=APP/'.local/compact-scale-v1';FEATURE=APP/'.local/compact-feature-v1';TARGETED=APP/'.local/compact-targeted-v1';EXPERT=APP/'.local/mouth-expert-v1'
WORK=Path(os.environ['A2F_RESEARCH_WORK']);TEACHER=Path(os.environ['A2F_TEACHER_ROOT']);ASSETS=WORK/'v3-model';BASELINE=FEATURE/'control/model.onnx'
EXPECTED='ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da'
sys.path.insert(0,str(OUT/'vendor-deps'));sys.path.insert(0,str(EXPERT/'vendor/syncnet'));sys.path.insert(0,str(EXPERT/'vendor-deps'));sys.path.insert(0,str(APP/'scripts/research/data_scale'));sys.path.insert(0,str(TEACHER))
os.environ['MLX_ENABLE_TF32']='0';os.environ['A2F_SCALE_OUTPUT']=str(SCALE)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(p,v):
 raw=json.dumps(v,indent=2).encode() if isinstance(v,(dict,list)) else v
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=raw:raise ValueError('Frozen artifact changed: '+str(p))
 if not p.exists():p.write_bytes(raw)

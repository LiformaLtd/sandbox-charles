from pathlib import Path
import hashlib,json,os,sys
APP=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('A2F_HISTORY_OUTPUT',APP/'.local/teacher-history-v1')).resolve()
WORK=Path(os.environ['A2F_RESEARCH_WORK']).resolve();TEACHER=Path(os.environ['A2F_TEACHER_ROOT']).resolve();ASSETS=WORK/'v3-model'
SCALE=APP/'.local/compact-scale-v1';BASELINE=APP/'.local/compact-feature-v1/control/model.onnx'
EXPECTED='ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da'
os.environ['MLX_ENABLE_TF32']='0'
sys.path.insert(0,str(APP/'scripts/research/lip_surface'))
sys.path.insert(0,str(TEACHER))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(memoryview(a).cast('B')).hexdigest()
def freeze(p,v):
 raw=json.dumps(v,indent=2).encode() if isinstance(v,(dict,list)) else v
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=raw:raise ValueError('Changed frozen artifact: '+str(p))
 if not p.exists():p.write_bytes(raw)

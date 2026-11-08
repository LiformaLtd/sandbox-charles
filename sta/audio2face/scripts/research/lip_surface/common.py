from pathlib import Path
import hashlib,json,os,sys
APP=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('A2F_SURFACE_OUTPUT',APP/'.local/compact-surface-v1')).resolve()
SCALE=APP/'.local/compact-scale-v1';FEATURE=APP/'.local/compact-feature-v1';TARGETED=APP/'.local/compact-targeted-v1'
WORK=Path(os.environ['A2F_RESEARCH_WORK']).resolve();BASELINE=FEATURE/'control/model.onnx'
TEACHER=Path(os.environ['A2F_TEACHER_ROOT']).resolve();ASSETS=WORK/'v3-model'
EXPECTED='ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da'
os.environ['MLX_ENABLE_TF32']='0';os.environ['A2F_SCALE_OUTPUT']=str(OUT)
sys.path.insert(0,str(APP/'scripts/research/data_scale'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(p,v):
 raw=json.dumps(v,indent=2).encode() if isinstance(v,(dict,list)) else v
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=raw:raise ValueError('Changed frozen artifact: '+str(p))
 if not p.exists():p.write_bytes(raw)
def inputs():
 freeze(OUT/'sources/model.py',(SCALE/'sources/model.py').read_bytes())
 for name in ['old-train','new-train','common-dev','replay-train','extra-train','replay-dev']:
  p=OUT/'features'/name;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():p.symlink_to(SCALE/'features'/name,target_is_directory=True)

from pathlib import Path
import os,sys,json,hashlib
APP=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=APP/'.local/mouth-expert-v1';VENDOR=OUT/'vendor';CORPUS=APP/'.local/compact-loss-v1/confirmation/variants'
WORK=Path(os.environ['A2F_RESEARCH_WORK']);BASELINE=APP/'.local/compact-feature-v1/control/model.onnx'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(p,v):
 raw=json.dumps(v,indent=2).encode() if isinstance(v,(dict,list)) else v
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=raw:raise ValueError('Changed frozen artifact: '+str(p))
 if not p.exists():p.write_bytes(raw)

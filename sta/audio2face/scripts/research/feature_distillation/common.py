from pathlib import Path
import hashlib,json,os,sys
APP=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('A2F_FEATURE_OUTPUT',APP/'.local/compact-feature-v1')).resolve()
SCALE=APP/'.local/compact-scale-v1'
WORK=Path(os.environ['A2F_RESEARCH_WORK']).resolve()
BASELINE=SCALE/'expanded-long/model.onnx'
EXPECTED='407739196cd98c89ec619b6af527f1021a3702cf8b1213c006f86408d1701137'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(p,v):
 raw=json.dumps(v,indent=2).encode() if isinstance(v,(dict,list)) else v
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=raw:raise ValueError('Changed frozen artifact: '+str(p))
 if not p.exists():p.write_bytes(raw)
sys.path.insert(0,str(APP/'scripts/research/data_scale'))

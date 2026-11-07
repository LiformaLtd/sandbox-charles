from pathlib import Path
import importlib.util,json,os,sys,time,hashlib
import numpy as np
os.environ['MLX_ENABLE_TF32']='0'
APP=Path(__file__).resolve().parents[3]
H=Path(os.environ.get('A2F_ABLATION_OUTPUT',APP/'.local/compact-followup')).resolve()
WORK=Path(os.environ['A2F_RESEARCH_WORK']).resolve()
RUNNER=Path(os.environ['A2F_TEACHER_ROOT']).resolve()
sys.path.insert(0,str(RUNNER))
from app.audio2face_v3 import animate,load_backend
path=Path(os.environ.get('A2F_DICT_ROOT',APP.parent/'audio2face-dict'))/'scripts/label.py'
spec=importlib.util.spec_from_file_location('fit',path);label=importlib.util.module_from_spec(spec);spec.loader.exec_module(label)
_,fit,_,gaps=label.prepare_fit(Path(os.environ.get('A2F_ASSET_ROOT',WORK/'v3-model')))
backend=load_backend(RUNNER/'models_cache/audio2face-v3/network.onnx','mlx')
rows=json.loads((H/'confirmation/manifest.json').read_text())
for i,row in enumerate(rows):
 target=H/'confirmation/labels'/f'{row["stem"]}.npz'
 if not target.exists():
  pcm=np.fromfile(row['audio'],'<f4');weights=[];truth=[]
  assert hashlib.sha256(Path(row['audio']).read_bytes()).hexdigest()==row['sha256_audio']
  for indices,geometry in animate(pcm,backend,identity='James',seed=0):
   weights.append(fit(geometry));truth.append(gaps(geometry))
  w,g=map(np.concatenate,(weights,truth));assert len(w)==(len(pcm)*60+15999)//16000
  assert np.isfinite(w).all() and np.isfinite(g).all()
  np.savez_compressed(target,weights=w,gaps=g)
 if i%8==0:print('Labeled',i+1,'of',len(rows),flush=True)

"""Offline-only compact projection; original graph/state/timestamps are unchanged."""
import importlib.util,numpy as np
from common import *
from geometry import prepare,PAIRS
sys.path.insert(0,str(TEACHER))
from app.audio2face_v3 import animate,load_backend,verify_model,MODEL_SHA256

def backend():
 import mlx.core as mx
 rig=prepare();ids=mx.array(rig['ids']);basis=mx.array(rig['basis']);offset=mx.array(rig['offset']);pairs=mx.array(PAIRS);skin=rig['skin'];baseline=mx.array(skin[PAIRS[:,0],1]-skin[PAIRS[:,1],1])
 def project(prediction):
  skin_delta=prediction[...,:72006].reshape(1,60,24002,3);target=mx.take(skin_delta,ids,axis=2)+offset
  q=mx.stack([target[...,a]@basis[:,:,a].T/len(rig['ids']) for a in range(3)],axis=2).reshape(1,60,117)
  norm=mx.mean(target*target,axis=2)
  upper=mx.take(skin_delta[...,1],pairs[:,0],axis=2);lower=mx.take(skin_delta[...,1],pairs[:,1],axis=2);gaps=baseline+upper-lower
  return mx.concatenate((q,norm,gaps),axis=-1)
 verify_model(TEACHER/'models_cache/audio2face-v3/network.onnx')
 source=(TEACHER/'app/audio2face_v3/mlx_backend.py').read_text();old='return _host(prediction), _host(state)';assert source.count(old)==1
 source=source.replace(old,'return _host(surface_project(prediction)), _host(state)')
 path=OUT/'sources/mlx_surface_projection.py';freeze(path,source.encode())
 spec=importlib.util.spec_from_file_location('app.audio2face_v3.surface_projection_backend',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.surface_project=project
 return module.mlx_backend(TEACHER/'models_cache/audio2face-v3/network.onnx')

def verify(row):
 from geometry import statistics
 rig=prepare();pcm=np.fromfile(row['audio'],'<f4')[:12000];reference=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx');wrapped=backend()
 truth=np.concatenate([g for _,g in animate(pcm,reference,identity='James',seed=0)]);compact=np.concatenate([g for _,g in animate(pcm,wrapped,identity='James',seed=0)])
 expected=statistics(truth,rig);np.testing.assert_allclose(compact[:,:120],expected,atol=2e-5,rtol=2e-5)
 xyz=truth[:,:72006].reshape(-1,24002,3);gaps=rig['skin'][PAIRS[:,0],1]-rig['skin'][PAIRS[:,1],1]+xyz[:,PAIRS[:,0],1]-xyz[:,PAIRS[:,1],1]
 np.testing.assert_allclose(compact[:,120:],gaps,atol=2e-5,rtol=2e-5)
 return dict(teacher_sha256=MODEL_SHA256,projection_max_abs_delta=float(np.max(abs(compact[:,:120]-expected))),gap_max_abs_delta=float(np.max(abs(compact[:,120:]-gaps))),geometry_values=truth.size,projector_sha256=sha(Path(__file__)))

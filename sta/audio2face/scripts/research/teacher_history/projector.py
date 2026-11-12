"""Expose actual dense mouth vertices; do not change the graph or recurrent state."""
import importlib.util,numpy as np
from common import *
from geometry import prepare,PAIRS
from app.audio2face_v3 import animate,load_backend,verify_model,MODEL_SHA256

def backend():
 import mlx.core as mx
 rig=prepare();ids=mx.array(rig['ids']);pairs=mx.array(PAIRS);skin=rig['skin'];base=mx.array(skin[PAIRS[:,0],1]-skin[PAIRS[:,1],1])
 def project(prediction):
  d=prediction[...,:72006].reshape(1,60,24002,3)
  dense=mx.take(d,ids,axis=2).reshape(1,60,6801)
  gaps=base+mx.take(d[...,1],pairs[:,0],axis=2)-mx.take(d[...,1],pairs[:,1],axis=2)
  return mx.concatenate((dense,gaps),axis=-1)
 verify_model(TEACHER/'models_cache/audio2face-v3/network.onnx')
 source=(TEACHER/'app/audio2face_v3/mlx_backend.py').read_text();old='return _host(prediction), _host(state)';assert source.count(old)==1
 source=source.replace(old,'return _host(mouth_project(prediction)), _host(state)')
 path=OUT/'sources/mlx_mouth_projection.py';freeze(path,source.encode())
 spec=importlib.util.spec_from_file_location('app.audio2face_v3.history_mouth_projection',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.mouth_project=project
 return module.mlx_backend(TEACHER/'models_cache/audio2face-v3/network.onnx')

def verify(row):
 rig=prepare();pcm=np.fromfile(row['audio']['cold']['path'],'<f4')[:12000]
 raw=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx');projected=backend()
 truth=np.concatenate([g for _,g in animate(pcm,raw,identity='James',seed=0)]);values=np.concatenate([g for _,g in animate(pcm,projected,identity='James',seed=0)])
 xyz=truth[:,:72006].reshape(-1,24002,3);expected=xyz[:,rig['ids']].reshape(len(xyz),6801);gap=rig['skin'][PAIRS[:,0],1]-rig['skin'][PAIRS[:,1],1]+xyz[:,PAIRS[:,0],1]-xyz[:,PAIRS[:,1],1]
 np.testing.assert_array_equal(values[:,:6801],expected);np.testing.assert_array_equal(values[:,6801:],gap)
 return dict(teacher_sha256=MODEL_SHA256,geometry_values=truth.size,surface_max_delta=float(np.max(abs(values[:,:6801]-expected))),gap_max_delta=float(np.max(abs(values[:,6801:]-gap))),source_sha256=sha(__file__))

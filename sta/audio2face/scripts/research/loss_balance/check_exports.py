"""Exact unchanged graph/constant tensors plus full waveform MLX/ORT checks."""
from common import *
import numpy as np,onnx,onnxruntime as ort,mlx.core as mx
from onnx import numpy_helper
from mlx_backend import Student
from inference import infer

def main():
 report={};base=onnx.load(BASELINE);basevalues={v.name:numpy_helper.to_array(v) for v in base.graph.initializer};ops=lambda model:[(n.op_type,list(n.input),list(n.output)) for n in model.graph.node]
 cases=json.loads((OUT/'browser/parity/parity-cases.json').read_text());rig=np.load(WORK/'arkit-student/rig-metrics.npz')
 for arm in ['control','balanced']:
  path=OUT/arm/'model.onnx';graph=onnx.load(path);assert ops(graph)==ops(base);values={v.name:numpy_helper.to_array(v).copy() for v in graph.graph.initializer}
  for k in ['filters','mel','student.mean','student.scale','student.left','student.fraction','student.channel_map']:np.testing.assert_array_equal(values[k],basevalues[k])
  assert path.stat().st_size==2251827;net=Student(values);opt=ort.SessionOptions();opt.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider']);detail=[]
  for case in cases:
   audio=np.fromfile(OUT/'browser/parity'/case['audio'],'<f4');waves=[];masks=[];offsets=np.arange(15,45)*16000//60
   for start in range(-8000,len(audio)-4000,8000):
    x=np.zeros(16000,np.float32);lo,hi=max(start,0),min(start+16000,len(audio));x[lo-start:hi-start]=audio[lo:hi];waves.append(x);masks.append((start+offsets>=0)&(start+offsets<len(audio)))
   wave=np.stack(waves);windows=np.lib.stride_tricks.sliding_window_view(np.pad(wave,((0,0),(200,200))),400,axis=1)[:,::160];fft=np.fft.rfft(windows*values['filters'][0,0],n=512,axis=-1);power=(fft.real**2+fft.imag**2).astype(np.float32);mel=np.log1p(np.einsum('btf,mf->btm',power,values['mel'][:,:,0],optimize=False)*10).transpose(0,2,1)
   pred=np.concatenate([np.array(mx.clip(net(mx.array(part))[:,15:45],0,1)) for part in np.array_split(mel,max(1,(len(mel)+63)//64))]);expected=np.zeros((sum(int(m.sum()) for m in masks),52),np.float32);expected[:,rig['active']]=pred[np.array(masks)];actual=infer(session,audio);delta=float(np.abs(expected-actual).max());assert delta<5e-5,(arm,case['name'],delta);detail.append(dict(case=case['name'],max_coefficient_delta=delta,frames=len(actual)))
  report[arm]=dict(graph_identical=True,frontend_identical=True,bytes=path.stat().st_size,cases=detail)
 freeze(OUT/'native-parity.json',report);print(json.dumps({a:max(v['max_coefficient_delta'] for v in r['cases']) for a,r in report.items()},indent=2))
if __name__=='__main__':main()

"""Compare selected MLX network against its waveform ONNX export on training audio."""
from common import *
import argparse,numpy as np,onnx,onnxruntime as ort
from onnx import numpy_helper
import mlx.core as mx
from model import Predictor

def main():
 p=argparse.ArgumentParser();p.add_argument('name');args=p.parse_args();run=OUT/args.name;lock()
 info=json.loads((run/'complete.json').read_text());width=192 if args.name.startswith('mentor') else 96
 values={t.name:numpy_helper.to_array(t).copy() for t in onnx.load(BASELINE).graph.initializer};net=Predictor(values,width)
 with np.load(run/'best.npz') as z:net.load_weights([(k,mx.array(z[k].transpose(0,2,1) if z[k].ndim==3 else z[k])) for k in z.files],strict=False)
 graph=onnx.load(run/'model.onnx');tensors={t.name:numpy_helper.to_array(t).copy() for t in graph.graph.initializer}
 filters=tensors['filters'];melbank=tensors['mel'][:,:,0];window=filters[0,0]
 options=ort.SessionOptions();options.intra_op_num_threads=2;session=ort.InferenceSession(str(run/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
 rows=[r for r in json.loads((SCALE/'datasets/old-train/manifest.json').read_text()) if r['split']=='train'][:3]
 errors=[]
 for row in rows:
  pcm=np.fromfile(row['audio'],'<f4');x=np.zeros(16000,np.float32);n=min(len(pcm),8000);x[8000:8000+n]=pcm[:n]
  windows=np.lib.stride_tricks.sliding_window_view(np.pad(x,(200,200)),400)[::160]
  fft=np.fft.rfft(windows*window,n=512,axis=-1);power=(fft.real**2+fft.imag**2).astype(np.float32)
  mel=np.log1p(np.einsum('tf,mf->mt',power,melbank,optimize=False)*10)[None]
  expected=session.run(None,{'pcm':x[None]})[0]
  actual=np.einsum('btc,ck->btk',np.array(mx.clip(net(mx.array(mel)),0,1)),tensors['student.channel_map'],optimize=False)
  if not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('Nonfinite parity output')
  errors.append(float(np.abs(expected-actual).max()))
 report=dict(model_sha256=sha(run/'model.onnx'),source_sha256=sha(__file__),training_clips=len(rows),max_coefficient_error=max(errors),passed=max(errors)<1e-5)
 write(run/'export-parity-v2.json',report);print(report,flush=True)
 if not report['passed']:raise ValueError('Export parity failed')
if __name__=='__main__':main()

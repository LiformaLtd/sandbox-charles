"""Native learned frontend/predictor versus unchanged-topology waveform ONNX export."""
from common import *
import argparse,numpy as np,onnx,onnxruntime as ort,mlx.core as mx
from onnx import numpy_helper
from mlx.utils import tree_unflatten
from model import Model

def main():
 p=argparse.ArgumentParser();p.add_argument('name');args=p.parse_args();lock();run=OUT/args.name
 values={t.name:numpy_helper.to_array(t).copy() for t in onnx.load(BASELINE).graph.initializer}
 net=Model(values,args.name.startswith('learned'))
 with np.load(run/'best.npz') as z:net.update(tree_unflatten([(k,mx.array(z[k])) for k in z.files]))
 options=ort.SessionOptions();options.intra_op_num_threads=2;session=ort.InferenceSession(str(run/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
 rows=[r for r in json.loads((SCALE/'datasets/old-train/manifest.json').read_text()) if r['split']=='train'][:3];errors=[]
 for row in rows:
  pcm=np.fromfile(row['audio'],'<f4');x=np.zeros(16000,np.float32);n=min(len(pcm),8000);x[8000:8000+n]=pcm[:n]
  windows=np.lib.stride_tricks.sliding_window_view(np.pad(x,(200,200)),400)[::160]
  fft=np.fft.rfft(windows*values['filters'][0,0],n=512,axis=-1);power=(fft.real**2+fft.imag**2).astype(np.float32).T[None]
  actual=np.einsum('btc,ck->btk',np.array(mx.clip(net(mx.array(power)),0,1)),values['student.channel_map'],optimize=False)
  expected=session.run(None,{'pcm':x[None]})[0]
  if not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('Nonfinite parity output')
  errors.append(float(np.abs(actual-expected).max()))
 report=dict(model_sha256=sha(run/'model.onnx'),source_sha256=sha(__file__),training_clips=3,max_coefficient_error=max(errors),passed=max(errors)<1e-5)
 write(run/'export-parity.json',report);print(report,flush=True)
 if not report['passed']:raise ValueError('Export parity failed')
if __name__=='__main__':main()

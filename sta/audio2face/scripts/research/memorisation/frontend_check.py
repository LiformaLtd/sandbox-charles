"""Localize a coefficient-level export tolerance miss without changing weights."""
from common import *
import numpy as np,onnx,onnxruntime as ort,mlx.core as mx
from onnx import numpy_helper,helper,TensorProto
from mlx_backend import Student

def main():
 base={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};rows=json.loads((OUT/'manifest.json').read_text());z=np.load(OUT/'batch.npz');waves=[]
 for row in rows:
  pcm=np.fromfile(row['audio'],'<f4')
  for start in range(-8000,len(pcm)-4000,8000):
   x=np.zeros(16000,np.float32);lo,hi=max(start,0),min(start+16000,len(pcm));x[lo-start:hi-start]=pcm[lo:hi];waves.append(x)
 report={}
 for arm in ['mixed','fit-mse']:
  graph=onnx.load(OUT/arm/'model.onnx');name=next(n.input[0] for n in graph.graph.node if n.name=='/student/encoder/Conv');del graph.graph.output[:];graph.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,64,101]));opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.log_severity_level=4;frontend=ort.InferenceSession(graph.SerializeToString(),sess_options=opt,providers=['CPUExecutionProvider']);mel=np.concatenate([frontend.run(None,{'pcm':x[None]})[0] for x in waves]);net=Student(base)
  with np.load(OUT/arm/'best.npz') as w:net.load_weights([(k,mx.array(w[k].transpose(0,2,1).copy() if w[k].ndim==3 else w[k])) for k in w.files],strict=False)
  cached=np.array(mx.clip(net(mx.array(z['mel']))[:,15:45],0,1));native=np.array(mx.clip(net(mx.array(mel))[:,15:45],0,1));mask=z['valid'];report[arm]=dict(frontend_max_abs_delta=float(np.abs(mel-z['mel']).max()),frontend_mean_abs_delta=float(np.abs(mel-z['mel']).mean()),mlx_output_change_from_frontend_only=float(np.abs(native[mask]-cached[mask]).max()))
 freeze(OUT/'frontend-check.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()

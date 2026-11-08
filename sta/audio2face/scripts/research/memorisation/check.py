"""Audit frozen inputs and complete exported waveform inference on all eight clips."""
from common import *
import numpy as np,onnx,onnxruntime as ort
from onnx import numpy_helper
sys.path.insert(0,str(APP/'scripts/research/compact_ablation'))
from evaluate import infer

def main():
 protocol=json.loads((OUT/'protocol.json').read_text());assert sha(BASELINE)==EXPECTED
 for name,digest in protocol['source_sha256'].items():assert sha(APP/name)==digest,name
 for name,digest in protocol['inputs_sha256'].items():assert sha(name)==digest,name
 assert sha(OUT/'batch.npz')==protocol['batch_sha256'] and sha(OUT/'manifest.json')==protocol['manifest_sha256']
 rows=json.loads((OUT/'manifest.json').read_text());assert len(rows)==len({r['text'] for r in rows})==8
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=rig['gap_basis'];rest=rig['rest']+rig['gap_offset'];base={v.name:numpy_helper.to_array(v) for v in onnx.load(BASELINE).graph.initializer};result={}
 for arm in ['mixed','fit-mse']:
  export=json.loads((OUT/arm/'export.json').read_text());assert sha(OUT/arm/'model.onnx')==export['model_sha256'];assert export['bytes']==2251827 and export['parameters']==331047
  graph={v.name:numpy_helper.to_array(v) for v in onnx.load(OUT/arm/'model.onnx').graph.initializer}
  for k in ['filters','mel','student.mean','student.scale','student.left','student.fraction','student.channel_map']:np.testing.assert_array_equal(graph[k],base[k])
  opt=ort.SessionOptions();opt.intra_op_num_threads=2;session=ort.InferenceSession(str(OUT/arm/'model.onnx'),sess_options=opt,providers=['CPUExecutionProvider']);delta=[];raw=[];ref=[];largest=0
  from mlx_backend import Student
  import mlx.core as mx
  net=Student(base)
  with np.load(OUT/arm/'best.npz') as z:
   # Checkpoint keys use PyTorch convolution layout for export.
   net.load_weights([(k,mx.array(z[k].transpose(0,2,1).copy() if z[k].ndim==3 else z[k])) for k in z.files],strict=False)
  with np.load(OUT/'batch.npz') as z:coeff=np.array(mx.clip(net(mx.array(z['mel']))[:,15:45],0,1));mask=z['valid']
  cursor=0
  for row in rows:
   pcm=np.fromfile(row['audio'],'<f4');pred=infer(session,pcm);n=row['windows'];expected=np.zeros((int(mask[cursor:cursor+n].sum()),52),np.float32);expected[:,rig['active']]=coeff[cursor:cursor+n][mask[cursor:cursor+n]];largest=max(largest,float(np.abs(pred-expected).max()));assert pred.shape==expected.shape and np.isfinite(pred).all()
   with np.load(SCALE/'datasets/old-train/labels'/(row['stem']+'.npz')) as z:truth=z['gaps'];target=z['weights']@basis+rest
   gaps=pred@basis+rest;raw.append(gaps-truth);delta.append(gaps-target);ref.append(target-truth);cursor+=n
  r=dict(raw_rms_mm=float(10*np.sqrt(np.square(np.concatenate(raw)).mean())),fit_rms_mm=float(10*np.sqrt(np.square(np.concatenate(delta)).mean())),reference_rms_mm=float(10*np.sqrt(np.square(np.concatenate(ref)).mean())),max_coefficient_delta=largest,strict_coefficient_parity_5e_minus5=bool(largest<5e-5))
  print(arm,r,flush=True);selected=json.loads((OUT/arm/'result.json').read_text())['best'];assert abs(r['raw_rms_mm']-selected['student_to_raw_rms_mm'])<1e-4;assert abs(r['fit_rms_mm']-selected['student_to_fit_rms_mm'])<1e-4
  result[arm]=r
 freeze(OUT/'export-check.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

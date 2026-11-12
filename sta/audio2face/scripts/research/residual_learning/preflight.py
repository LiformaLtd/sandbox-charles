"""Freeze data, matched initial functions, head budget and waveform parity."""
from common import *
import numpy as np,onnx,onnxruntime as ort,mlx.core as mx,mlx.nn as nn
from onnx import numpy_helper
from mlx.utils import tree_flatten
from train_heads import datasets,sample,constants,evaluate
from residual_model import Student,Head,features,export
from mlx_backend import objective

def main():
    values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};base=Student(values);base.freeze();train,replay,groups,dev=datasets();active,basis,rest,lips=constants();rng=np.random.default_rng(9037);square=np.zeros(96,np.float64);count=0
    for _ in range(32):
        b,_=sample(train,replay,groups,rng);x=b['mel'][:,15:45][b['valid']];square+=np.square(x).sum(0);count+=len(x)
    scale=np.maximum(np.sqrt(square/count),.1).astype(np.float32);path=OUT/'feature-scale.npy'
    if path.exists():np.testing.assert_array_equal(np.load(path),scale)
    else:np.save(path,scale)
    batch,_=sample(train,replay,groups,np.random.default_rng(1037));h=mx.array(batch['mel']);expected=(h@mx.array(values['student.output.weight'][:,:,0]).T+mx.array(values['student.output.bias']))*base.scale+base.mean;report={}
    for arm in ['control','residual']:
        head=Head(values,scale,arm);delta=float(mx.max(mx.abs(head(h)-expected)));assert delta<5e-5,delta
        args=[mx.array(batch[k]) for k in ['mel','weights','gaps','valid']]+[basis,rest,lips];loss,grads=nn.value_and_grad(head,objective)(head,*args);assert all(np.isfinite(np.array(g)).all() for _,g in tree_flatten(grads))
        dest=OUT/'preflight'/(arm+'.onnx');result=export(head,values,active,dest);option=ort.SessionOptions();option.intra_op_num_threads=2;session=ort.InferenceSession(str(dest),sess_options=option,providers=['CPUExecutionProvider']);original=ort.InferenceSession(str(BASELINE),sess_options=option,providers=['CPUExecutionProvider']);pcm=np.random.default_rng(37).normal(0,.08,(1,16000)).astype(np.float32)
        error=float(np.max(np.abs(session.run(None,{'pcm':pcm})[0]-original.run(None,{'pcm':pcm})[0])));assert error<5e-5,error
        report[arm]=dict(**result,initial_feature_delta=delta,initial_waveform_delta=error,loss=float(loss),development=evaluate(head,dev,basis,rest))
    assert report['control']['bytes']==report['residual']['bytes'];assert abs(report['control']['loss']-report['residual']['loss'])<5e-5
    freeze(OUT/'preflight.json',report)
    sources=[HERE/n for n in ['common.py','residual_model.py','cache_features.py','train_heads.py','preflight.py']]+[APP/'scripts/research/data_scale'/n for n in ['mlx_backend.py','features.py']]+[SCALE/'sources/model.py']
    caches={p.parent.name:json.loads(p.read_text()) for p in (OUT/'features').glob('*/complete.json')}
    for name,meta in caches.items():assert sha(OUT/'features'/name/'hidden.npy')==meta['sha256']
    freeze(OUT/'training-protocol.json',dict(sources={str(p.relative_to(APP)):sha(p) for p in sources},baseline_sha256=sha(BASELINE),feature_scale_sha256=sha(path),caches=caches,cache_protocol_sha256=sha(OUT/'features/protocol.json'),design_sha256=sha(APP/'docs/plans/compact-residual-learning.md'),confirmation_planned_sha256=sha(OUT/'confirmation/base/planned.json'),seed=37,updates=7440,head_parameters=16839))
    print('Equal initial functions/losses, frozen base, head sizes, waveform exports and cache locks pass.',flush=True)

if __name__=='__main__':main()

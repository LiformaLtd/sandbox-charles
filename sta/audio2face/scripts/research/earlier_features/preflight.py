"""Check matched heads, feature/prediction alignment and waveform export locks."""
from common import *
import numpy as np,onnx,onnxruntime as ort,mlx.core as mx,mlx.nn as nn
from onnx import numpy_helper
from mlx.utils import tree_flatten
from train_heads import datasets,sample,constants,evaluate
from residual_model import Student,Head,export
from mlx_backend import objective

def main():
    values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};active,basis,rest,lips=constants();report={};samples={}
    for arm in ['late','early']:
        train,replay,groups,dev=datasets(arm);rng=np.random.default_rng(9037);square=np.zeros(96,np.float64);count=0
        for _ in range(32):
            b,_=sample(train,replay,groups,rng);x=b['mel'][:,15:45,:96][b['valid']];square+=np.square(x).sum(0);count+=len(x)
        scale=np.maximum(np.sqrt(square/count),.1).astype(np.float32);path=OUT/('feature-scale-'+arm+'.npy')
        if path.exists():np.testing.assert_array_equal(np.load(path),scale)
        else:np.save(path,scale)
        batch,indices=sample(train,replay,groups,np.random.default_rng(1037));samples[arm]=(batch,indices);head=Head(values,scale,arm);h=mx.array(batch['mel']);expected=h[:,:,96:]*head.scale+head.mean
        delta=float(mx.max(mx.abs(head(h)-expected)));assert delta<5e-5,delta
        loss,grads=nn.value_and_grad(head,objective)(head,*[mx.array(batch[k]) for k in ['mel','weights','gaps','valid']],basis,rest,lips);assert all(np.isfinite(np.array(g)).all() for _,g in tree_flatten(grads))
        dest=OUT/'preflight'/(arm+'.onnx');result=export(head,values,active,dest);option=ort.SessionOptions();option.intra_op_num_threads=2;session=ort.InferenceSession(str(dest),sess_options=option,providers=['CPUExecutionProvider']);original=ort.InferenceSession(str(BASELINE),sess_options=option,providers=['CPUExecutionProvider']);pcm=np.random.default_rng(37).normal(0,.08,(1,16000)).astype(np.float32)
        error=float(np.max(np.abs(session.run(None,{'pcm':pcm})[0]-original.run(None,{'pcm':pcm})[0])));assert error<5e-5,error
        report[arm]=dict(**result,initial_feature_delta=delta,initial_waveform_delta=error,loss=float(loss),development=evaluate(head,dev,basis,rest))
    assert samples['late'][1]==samples['early'][1];np.testing.assert_array_equal(samples['late'][0]['mel'][:,:,96:],samples['early'][0]['mel'][:,:,96:])
    assert report['late']['bytes']==report['early']['bytes'];assert abs(report['late']['loss']-report['early']['loss'])<5e-5
    freeze(OUT/'preflight.json',report)
    sources=[HERE/n for n in ['common.py','residual_model.py','cache_features.py','train_heads.py','preflight.py']]+[APP/'scripts/research/data_scale'/n for n in ['mlx_backend.py','features.py']]+[SCALE/'sources/model.py']
    caches={p.parent.name:json.loads(p.read_text()) for p in (OUT/'features').glob('*/complete.json')}
    for name,meta in caches.items():
        assert sha(OUT/'features'/name/'early.npy')==meta['early_sha256'];assert sha(OUT/'features'/name/'base-p.npy')==meta['prediction_sha256'];assert sha(PREVIOUS/'features'/name/'hidden.npy')==meta['late_sha256']
    freeze(OUT/'training-protocol.json',dict(sources={str(p.relative_to(APP)):sha(p) for p in sources},baseline_sha256=sha(BASELINE),normalization_sha256={a:sha(OUT/('feature-scale-'+a+'.npy')) for a in report},caches=caches,cache_protocol_sha256=sha(OUT/'features/protocol.json'),design_sha256=sha(APP/'docs/plans/compact-earlier-features.md'),confirmation_planned_sha256=sha(OUT/'confirmation/base/planned.json'),seed=37,updates=7440,head_parameters=16839))
    print('Matched initial functions/losses, shared prediction cache, frozen base, exports and cache locks pass.',flush=True)

if __name__=='__main__':main()

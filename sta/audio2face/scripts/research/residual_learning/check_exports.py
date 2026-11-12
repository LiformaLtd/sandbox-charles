"""Frozen base tensors, matched graphs, waveform native/ORT and inactive channels."""
from common import *
import numpy as np,onnx,onnxruntime as ort,mlx.core as mx
from onnx import numpy_helper
from residual_model import Student,Head,features
sys.path.insert(0,str(APP/'scripts/research/closure_protected'))
from inference import infer

def main():
    v={t.name:numpy_helper.to_array(t).copy() for t in onnx.load(BASELINE).graph.initializer};base=Student(v);base.freeze();active=np.load(WORK/'arkit-student/rig-metrics.npz')['active'];scale=np.load(OUT/'feature-scale.npy');cases=json.loads((OUT/'browser/parity/parity-cases.json').read_text());reports={};ops=lambda g:[(n.op_type,list(n.input),list(n.output)) for n in g.graph.node];reference=onnx.load(OUT/'preflight/control.onnx')
    for arm in ['control','residual']:
        path=OUT/arm/'model.onnx';graph=onnx.load(path);assert ops(graph)==ops(reference);tensors={t.name:numpy_helper.to_array(t) for t in graph.graph.initializer}
        for name,value in v.items():
            if not name.startswith('onnx::'):np.testing.assert_array_equal(tensors[name],value,err_msg=name)
            else:assert any(t.dtype==value.dtype and t.shape==value.shape and np.array_equal(t,value) for n,t in tensors.items() if n.startswith('onnx::'))
        assert path.stat().st_size==2321912;head=Head(v,scale,arm);head.restore(OUT/arm/'best.npz');option=ort.SessionOptions();option.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=option,providers=['CPUExecutionProvider']);detail=[]
        for case in cases:
            audio=np.fromfile(OUT/'browser/parity'/case['audio'],'<f4');waves=[];masks=[];offsets=np.arange(15,45)*16000//60
            for start in range(-8000,len(audio)-4000,8000):
                pcm=np.zeros(16000,np.float32);lo,hi=max(start,0),min(start+16000,len(audio));pcm[lo-start:hi-start]=audio[lo:hi];waves.append(pcm);masks.append((start+offsets>=0)&(start+offsets<len(audio)))
            wave=np.stack(waves);windows=np.lib.stride_tricks.sliding_window_view(np.pad(wave,((0,0),(200,200))),400,axis=1)[:,::160];fft=np.fft.rfft(windows*v['filters'][0,0],n=512,axis=-1);power=(fft.real**2+fft.imag**2).astype(np.float32);mel=np.log1p(np.einsum('btf,mf->btm',power,v['mel'][:,:,0],optimize=False)*10).transpose(0,2,1)
            pred=np.concatenate([np.array(mx.clip(head(features(base,mx.array(part)))[:,15:45],0,1)) for part in np.array_split(mel,max(1,(len(mel)+63)//64))]);expected=np.zeros((sum(int(m.sum()) for m in masks),52),np.float32);expected[:,active]=pred[np.array(masks)];actual=infer(session,audio);delta=float(np.max(np.abs(actual-expected)));assert delta<5e-5,(arm,case['name'],delta);assert not actual[:,np.setdiff1d(np.arange(52),active)].any();detail.append(dict(case=case['name'],delta=delta,frames=len(actual)))
        reports[arm]=dict(frozen_base_identical=True,matched_graph=True,bytes=path.stat().st_size,cases=detail)
    freeze(OUT/'native-parity.json',reports);print(json.dumps({a:max(c['delta'] for c in r['cases']) for a,r in reports.items()},indent=2))

if __name__=='__main__':main()

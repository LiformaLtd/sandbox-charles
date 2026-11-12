"""Check exact widening, export parity and source/data locks before training."""
from common import *
import numpy as np,onnx,onnxruntime as ort,mlx.core as mx,mlx.nn as nn
from onnx import numpy_helper
from model import Assistant,Student,features,export
from train_assistant import data,constants,evaluate

def main():
    if not (OUT/'audit/result.json').exists():raise ValueError('Finish target audit first')
    assert sha(BASELINE)=='ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da'
    mx.set_default_device(mx.gpu);values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};tiny=Student(values);wide=Assistant(values);train,replay,groups,dev=data();active,basis,rest,lips=constants()
    b=train.batch(np.arange(8));x=mx.array(b['mel']);a=np.array(tiny(x));c=np.array(wide(x));error=float(np.max(np.abs(a-c)));assert error<5e-5,error
    feature_error=float(np.max(np.abs(np.array(features(tiny,x))-np.array(wide.feature(x)))));assert feature_error<5e-5,feature_error
    from mlx_backend import objective
    def loss(net):return objective(net,x,*[mx.array(b[k]) for k in ['weights','gaps','valid']],basis,rest,lips)
    _,gradient=nn.value_and_grad(wide,loss)(wide)
    encoder_gradient=np.array(gradient['encoder']['weight']);symmetry_delta=float(np.max(np.abs(encoder_gradient[:96]-encoder_gradient[96:])));assert symmetry_delta>1e-8,symmetry_delta
    root=OUT/'preflight';root.mkdir(exist_ok=True);reports={}
    for name,net in [('tiny',tiny),('assistant',wide)]:
        report=export(net,values,active,root/(name+'.onnx'));option=ort.SessionOptions();option.intra_op_num_threads=2;session=ort.InferenceSession(str(root/(name+'.onnx')),sess_options=option,providers=['CPUExecutionProvider'])
        wave=np.random.default_rng(37).normal(0,.08,(1,16000)).astype(np.float32)
        sys.path.insert(0,str(APP/'scripts/research/closure_protected'));from inference import infer
        # Match the waveform frontend separately, then all60 output frames.
        filters=values['filters'];windows=np.lib.stride_tricks.sliding_window_view(np.pad(wave,((0,0),(200,200))),400,axis=1)[:,::160];fft=np.fft.rfft(windows*filters[0,0],n=512,axis=-1);power=(fft.real**2+fft.imag**2).astype(np.float32);mel=np.log1p(np.einsum('btf,mf->btm',power,values['mel'][:,:,0],optimize=False)*10).transpose(0,2,1)
        expected=np.zeros((1,60,52),np.float32);expected[:,:,active]=np.array(mx.clip(net(mx.array(mel)),0,1));actual=session.run(None,{'pcm':wave})[0];delta=float(np.max(np.abs(expected-actual)));assert delta<5e-5,delta
        reports[name]=dict(**report,parity=delta)
    initial=evaluate(tiny,dev,basis,rest);freeze(OUT/'preflight.json',dict(widen_max_delta=error,feature_max_delta=feature_error,duplicate_gradient_difference=symmetry_delta,exports=reports,initial_dev=initial,audit_sha256=sha(OUT/'audit/result.json')))
    sources=[HERE/n for n in ['common.py','model.py','train_assistant.py','preflight.py']]+[APP/'scripts/research/data_scale'/n for n in ['features.py','mlx_backend.py','prepare_corpus.py']]+[SCALE/'sources/model.py']
    prior=json.loads((SCALE/'training-protocol.json').read_text());arrays=prior['array_sha256']
    for name,digest in arrays.items():assert sha(SCALE/name)==digest,name
    freeze(OUT/'training-protocol.json',dict(sources={str(p.relative_to(APP)):sha(p) for p in sources},baseline_sha256=sha(BASELINE),array_sha256=arrays,design_sha256=sha(APP/'docs/plans/compact-assistant-teacher.md'),audit_sha256=sha(OUT/'audit/result.json'),assistant_updates=7440,student_updates=3720,seed=37,feature_loss=.1,output_kd_loss=.25))
    print('Widening, features, waveform exports, dev reference and frozen data pass.',flush=True)

if __name__=='__main__':main()

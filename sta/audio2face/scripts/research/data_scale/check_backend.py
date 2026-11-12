"""Check native MLX forward, mixed loss and gradients against established CPU torch."""
import json,os,sys,time,importlib.util
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from prepare_corpus import OUT,WORK,APP,sha
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper
os.environ['MLX_ENABLE_TF32']='0'
import mlx.core as mx,mlx.nn as nn,mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_backend import Student,objective


def main():
    torch.set_num_threads(2);root=OUT/'memorization';protocol=json.loads((root/'protocol.json').read_text())
    spec=importlib.util.spec_from_file_location('backend_reference',root/'sources/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(protocol['baseline']).graph.initializer}
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];basis=rig['gap_basis'][active];rest=rig['rest']+rig['gap_offset']
    names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active],np.float32)
    source=next((root/'feature-cache/features').glob('*.npz'))
    with np.load(source) as z:arrays=[np.tile(z[str(i)],(3,)+(1,)*(z[str(i)].ndim-1))[:64].copy() for i in range(4)]
    net=base.Student(96,values['student.mean'],values['student.scale'],active)
    net.load_state_dict({k:torch.from_numpy(values['student.'+k].copy()) for k in net.state_dict()})
    x,y,g,m=[torch.from_numpy(v) for v in arrays];tb,tr,tl=[torch.from_numpy(v) for v in [basis,rest,lips]]
    pred=net(x)[:,15:45];w=m[...,None];count=m.sum();geom=pred.clamp(0,1)@tb+tr
    shape=(F.smooth_l1_loss((pred-y)/net.scale,torch.zeros_like(pred),reduction='none',beta=.5)*w*tl).sum()/(count*39)
    gap=(F.smooth_l1_loss(geom,g,reduction='none',beta=.14)*w).sum()/(count*9)
    adjacent=m[:,1:]&m[:,:-1];velocity=(((pred.diff(dim=1)-y.diff(dim=1)).abs()/net.scale)*adjacent[...,None]).sum()/(count*39)
    actual,estimate=g.median(-1).values,geom.median(-1).values;contact=actual<=.1
    closure=(F.binary_cross_entropy_with_logits((.1-estimate)/.16,contact.float(),reduction='none')*(1+contact)*m).sum()/count
    near=((actual[:,1:]<.35)|(actual[:,:-1]<.35))&adjacent
    timing=(F.smooth_l1_loss(estimate.diff(dim=1),actual.diff(dim=1),reduction='none',beta=.1)*near).sum()/near.sum().clamp(min=1)
    loss=shape+1.2*gap+.12*velocity+.22*closure+.45*timing;loss.backward()
    other=Student(values);inputs=[mx.array(v) for v in arrays];mb,mr,ml=[mx.array(v) for v in [basis,rest,lips]]
    fn=nn.value_and_grad(other,objective);value,gradients=fn(other,*inputs,mb,mr,ml);prediction=other(inputs[0])[:,15:45];mx.eval(value,gradients,prediction)
    grad={}
    for name,v in tree_flatten(gradients):
        a=np.array(v);grad[name]=a.transpose(0,2,1) if a.ndim==3 else a
    squared=0.;reference=0.
    for name,p in net.named_parameters():
        a=p.grad.numpy();b=grad[name];squared+=float(np.square(a-b).sum());reference+=float(np.square(a).sum())
    result=dict(mlx_version=mx.__version__,torch_version=torch.__version__,forward_max_delta=float(np.max(np.abs(np.array(prediction)-pred.detach().numpy()))),
                gap_max_delta_mm=float(10*np.max(np.abs(np.einsum('btc,ck->btk',np.array(prediction)-pred.detach().numpy(),basis,optimize=False)))),
                gap_rms_delta_mm=float(10*np.sqrt(np.square(np.einsum('btc,ck->btk',np.array(prediction)-pred.detach().numpy(),basis,optimize=False)).mean())),
                loss_cpu=float(loss.detach()),loss_mlx=float(value),gradient_relative_l2=float(np.sqrt(squared/reference)),
                source_sha256=sha(Path(__file__).with_name('mlx_backend.py')))
    if result['forward_max_delta']>2e-5 or result['gap_rms_delta_mm']>.01 or result['gap_max_delta_mm']>.05 or abs(result['loss_cpu']-result['loss_mlx'])>2e-5 or result['gradient_relative_l2']>.001:raise ValueError('Training backend failed parity: '+str(result))
    opt=optim.AdamW(learning_rate=.00012,weight_decay=.015,bias_correction=True)
    torch_optimizer=torch.optim.AdamW(net.parameters(),lr=.00012,weight_decay=.015)
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.);torch_optimizer.step()
    value,grads=fn(other,*inputs,mb,mr,ml);grads,_=optim.clip_grad_norm(grads,1.);opt.update(other,grads);mx.eval(other.parameters(),opt.state)
    updated=other.torch_weights()
    result['first_update_max_delta']=max(float(np.max(np.abs(p.detach().numpy()-updated[n]))) for n,p in net.named_parameters())
    if result['first_update_max_delta']>2e-5:raise ValueError('Optimizer update parity failed')
    for _ in range(3):
        value,grads=fn(other,*inputs,mb,mr,ml);grads,_=optim.clip_grad_norm(grads,1.);opt.update(other,grads);mx.eval(value,other.parameters(),opt.state)
    begin=time.monotonic()
    for _ in range(20):
        value,grads=fn(other,*inputs,mb,mr,ml);grads,_=optim.clip_grad_norm(grads,1.);opt.update(other,grads);mx.eval(value,other.parameters(),opt.state)
    result.update(seconds_per_update_batch64=(time.monotonic()-begin)/20,loss_after_updates=float(value))
    if result['loss_after_updates']>=result['loss_mlx']:raise ValueError('Loss did not descend')
    (OUT/'mlx-backend-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

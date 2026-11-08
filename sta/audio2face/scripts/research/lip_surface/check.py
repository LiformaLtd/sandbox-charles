"""Real-frame dense/quadratic and CPU/MLX gradient checks, before optimizer runs."""
import importlib.util,numpy as np,torch
from common import *
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper
import mlx.core as mx,mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_backend import Student
from geometry import prepare,statistics
from loss import surface_loss

def main():
 torch.set_num_threads(2);rig=prepare();row=json.loads((SCALE/'datasets/old-train/manifest.json').read_text())[0]
 from teacher import animate,load_backend
 run=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx');pcm=np.fromfile(row['audio'],'<f4')[:12000];raw=np.concatenate([g for _,g in animate(pcm,run,identity='James',seed=0)])[:5]
 target=raw[:,:72006].reshape(-1,24002,3)[:,rig['ids']]+rig['offset'];stats=statistics(raw,rig)
 weights=np.random.default_rng(37).uniform(.05,.8,size=(5,39)).astype(np.float32)
 w=torch.tensor(weights,requires_grad=True);dense=((torch.einsum('tc,cva->tva',w,torch.from_numpy(rig['basis']))-torch.from_numpy(target))**2).sum(-1).mean();dense.backward();grad_dense=w.grad.clone()
 w2=torch.tensor(weights,requires_grad=True);q=torch.from_numpy(stats[:,:117].reshape(-1,3,39));g=torch.from_numpy(rig['gram']);s=torch.from_numpy(stats[:,117:]);quadratic=(torch.einsum('ti,aij,tj->ta',w2,g,w2)-2*torch.einsum('tac,tc->ta',q,w2)+s).sum(-1).mean();quadratic.backward()
 dense_delta=abs(float(dense.detach())-float(quadratic.detach()));dense_grad=float((grad_dense-w2.grad).abs().max());assert dense_delta<2e-5 and dense_grad<2e-5
 values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Student(values)
 spec=importlib.util.spec_from_file_location('surface_cpu_reference',SCALE/'sources/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);cpu=base.Student(96,values['student.mean'],values['student.scale'],rig['active']);cpu.load_state_dict({k:torch.from_numpy(values['student.'+k].copy()) for k in cpu.state_dict()})
 x=np.load(SCALE/'features/old-train/mel.npy',mmap_mode='r')[:3].copy();valid=np.load(SCALE/'features/old-train/valid.npy',mmap_mode='r')[:3].copy()
 with np.load(OUT/'surface-tracks'/(row['sha256_audio']+'.npz')) as z:track=z['stats']
 y=np.zeros((3,30,120),np.float32);cursor=0
 for i,mask in enumerate(valid):count=int(mask.sum());y[i,mask]=track[cursor:cursor+count];cursor+=count
 pred=cpu(torch.from_numpy(x))[:,15:45].clamp(0,1);q=torch.from_numpy(y[...,:117].reshape(3,30,3,39));s=torch.from_numpy(y[...,117:]);squared=(torch.einsum('bti,aij,btj->bta',pred,g,pred)-2*torch.einsum('btac,btc->bta',q,pred)+s).sum(-1).clamp(min=0);reference=(squared*torch.from_numpy(valid)).sum()/valid.sum();reference.backward()
 value,gradient=nn.value_and_grad(net,surface_loss)(net,mx.array(x),mx.array(y),mx.array(valid),mx.array(rig['gram']));mx.eval(value,gradient)
 actual={k:np.array(v) for k,v in tree_flatten(gradient)};delta=total=0.
 for name,p in cpu.named_parameters():
  a=p.grad.numpy();b=actual[name];b=b.transpose(0,2,1) if b.ndim==3 else b;delta+=float(np.square(a-b).sum());total+=float(np.square(a).sum())
 relative=float(np.sqrt(delta/total));loss_delta=abs(float(value)-float(reference.detach()));assert relative<.001 and loss_delta<2e-5
 freeze(OUT/'surface-loss-check.json',dict(dense_quadratic_loss_delta=dense_delta,dense_quadratic_gradient_max_delta=dense_grad,cpu_mlx_loss_delta=loss_delta,cpu_mlx_gradient_relative_l2=relative,source_sha256=sha(__file__),loss_sha256=sha(HERE/'loss.py'),geometry_sha256=sha(HERE/'geometry.py')))
 print(json.dumps(json.loads((OUT/'surface-loss-check.json').read_text()),indent=2))
if __name__=='__main__':main()

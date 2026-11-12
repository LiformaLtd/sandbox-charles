"""Check the CPU→MLX cotangent bridge and fix a training-only loss scale."""
from common import *
if os.environ.get('A2F_TORCH_SITE_PACKAGES'):sys.path.append(os.environ['A2F_TORCH_SITE_PACKAGES'])
import numpy as np,torch,onnx,importlib.util
from onnx import numpy_helper
import mlx.core as mx,mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_backend import Student,objective
from features import Dataset
from browser_auxiliary import BrowserExpert

def norm(grads):return float(mx.sqrt(sum(mx.sum(g*g) for _,g in tree_flatten(grads))))
def main():
 torch.set_num_threads(8);mx.set_default_device(mx.gpu);values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Student(values);train=Dataset(['old-train','new-train']);replay=Dataset(['replay-train','extra-train']);voices=sorted(set(train.voices));groups=[np.flatnonzero(train.voices==v) for v in voices];rng=np.random.default_rng(1037)
 ix=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16);a,b=train.batch(ix),replay.batch(old);batch={k:np.concatenate((a[k],b[k])) for k in a};tensors=[mx.array(batch[k]) for k in ['mel','weights','gaps','valid']];rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];basis=mx.array(rig['gap_basis'][active]);rest=mx.array(rig['rest']+rig['gap_offset']);names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=mx.array(np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active],np.float32));base,baseg=nn.value_and_grad(net,objective)(net,*tensors,basis,rest,lips);base_norm=norm(baseg)
 expert=BrowserExpert();rows=[];first=None
 for voice in range(8):
  position=next(j for j in range(voice*6,voice*6+6) if expert.mapping[int(ix[j])]['eligible']);index=int(ix[position]);mel=mx.array(batch['mel'][position:position+1]);prediction=np.array(net(mel)[:,15:45])[0];loss,cot,detail=expert.gradient(prediction,batch['weights'][position],index,seed=100000+voice);cot=mx.array(cot[None]);fn=lambda n:(n(mel)[:,15:45]*cot).sum();_,grads=nn.value_and_grad(net,fn)(net);rows.append(dict(voice=voices[voice],index=index,position=position,loss=loss,backbone_norm=norm(grads),**detail))
  if first is None and detail['active']:first=(mel,cot,grads)
 if first is None:raise ValueError('No active training auxiliary gradients')
 mel,cot,mlgrads=first;spec=importlib.util.spec_from_file_location('frozen_export',SCALE/'sources/model.py');model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model);tnet=model.Student(96,values['student.mean'],values['student.scale'],active);state={k:torch.from_numpy(values['student.'+k].copy()) for k in tnet.state_dict()};tnet.load_state_dict(state);tnet.eval();tm=torch.tensor(np.array(mel));out=tnet(tm).transpose(1,2) if False else tnet(tm);expected=np.array(net(mel));delta=float(np.abs(out.detach().numpy()-expected).max());loss=(out[:,15:45]*torch.tensor(np.array(cot))).sum();loss.backward();tg=dict(tnet.named_parameters());diff=0.;total=0.;maxdelta=0.
 for name,g in tree_flatten(mlgrads):
  x=np.array(g);x=x.transpose(0,2,1) if x.ndim==3 else x;y=tg[name].grad.numpy();diff+=float(np.square(x-y).sum());total+=float(np.square(y).sum());maxdelta=max(maxdelta,float(np.abs(x-y).max()))
 relative=float(np.sqrt(diff/max(total,1e-20)));assert delta<5e-5 and relative<.01,(delta,relative)
 mean=float(np.mean([r['backbone_norm'] for r in rows]));coefficient=.25*base_norm/mean
 freeze(OUT/'calibration.json',dict(baseline_sha256=sha(BASELINE),source_sha256=sha(__file__),base_loss=float(base),base_backbone_norm=base_norm,auxiliary_mean_backbone_norm=mean,auxiliary_lambda=coefficient,ratio=.25,first_batch_indices=ix.tolist(),rows=rows,bridge=dict(max_output_delta=delta,relative_gradient_l2=relative,max_gradient_delta=maxdelta,passes=True),note='Training data only; first main batch at seed37. No model update or test score used.'))
 print((OUT/'calibration.json').read_text(),flush=True)
if __name__=='__main__':main()

"""CPU reference checks for training-only cosine hints and encoder gradient flow."""
from common import *
import importlib.util,numpy as np,torch
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper
from model import TrainingModel,cosine_loss,mx,nn
from mlx.utils import tree_flatten

def main():
 torch.set_num_threads(2);mx.random.seed(37)
 values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};assert sha(BASELINE)==EXPECTED
 net=TrainingModel(values)
 spec=importlib.util.spec_from_file_location('cpu_hint_reference',SCALE/'sources/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 active=np.load(WORK/'arkit-student/rig-metrics.npz')['active'];student=base.Student(96,values['student.mean'],values['student.scale'],active);student.load_state_dict({k:torch.from_numpy(values['student.'+k].copy()) for k in student.state_dict()})
 head=torch.nn.Linear(96,768);head.weight.data.copy_(torch.from_numpy(np.array(net.hint_head.weight)));head.bias.data.copy_(torch.from_numpy(np.array(net.hint_head.bias)))
 x=np.load(SCALE/'features/old-train/mel.npy',mmap_mode='r')[:6].copy();h=np.load(OUT/'hints/old-train/hints.npy',mmap_mode='r')[:6].astype(np.float32);m=np.load(SCALE/'features/old-train/valid.npy',mmap_mode='r')[:6].copy()
 tx=torch.from_numpy(x);hidden=student.blocks(torch.nn.functional.gelu(student.encoder(tx),approximate='tanh'));lo=hidden.index_select(2,student.left);hi=hidden.index_select(2,student.left+1);hidden=(lo+student.fraction*(hi-lo))[:,:,15:45].transpose(1,2);p=head(hidden);target=torch.from_numpy(h)
 norm=lambda a:a/torch.sqrt((a*a).sum(-1,keepdim=True)+1e-6)
 reference=((1-(norm(p)*norm(target)).sum(-1))*torch.from_numpy(m)).sum()/m.sum();reference.backward()
 def hint(n,x,h,m):return cosine_loss(n.hints(x),h,m)
 value,grads=nn.value_and_grad(net,hint)(net,mx.array(x),mx.array(h),mx.array(m));mx.eval(value,grads)
 actual={k:np.array(v) for k,v in tree_flatten(grads)};squared=total=0.
 for name,param in list(student.named_parameters())+[(f'hint_head.{n}',v) for n,v in head.named_parameters()]:
  if param.grad is None:continue
  key=name if name.startswith('hint_head.') else 'student.'+name
  v=actual[key];v=v.transpose(0,2,1) if v.ndim==3 else v
  a=param.grad.numpy();squared+=float(np.square(a-v).sum());total+=float(np.square(a).sum())
 relative=float(np.sqrt(squared/total));difference=abs(float(value)-float(reference.detach()));assert relative<.001 and difference<2e-5,(relative,difference)
 def zero(n,x,h,m):return mx.array(0.)*hint(n,x,h,m)
 _,zero_grads=nn.value_and_grad(net,zero)(net,mx.array(x),mx.array(h),mx.array(m));mx.eval(zero_grads)
 assert all(np.count_nonzero(np.array(v))==0 for _,v in tree_flatten(zero_grads))
 backbone_norm=float(np.sqrt(sum(np.square(v).sum() for k,v in actual.items() if k.startswith('student.'))));assert backbone_norm>0
 result=dict(loss_cpu=float(reference.detach()),loss_mlx=float(value),loss_difference=difference,gradient_relative_l2=relative,backbone_hint_gradient_norm=backbone_norm,control_hint_gradients_zero=True,model_sha256=sha(HERE/'model.py'),check_sha256=sha(__file__))
 freeze(OUT/'backend-check.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

"""Frozen training-only SyncNet and differentiable James coefficient loss."""
from common import *
import numpy as np,torch
from render import Renderer
from SyncNetModel import S

class ExpertLoss:
 def __init__(self):
  torch.set_num_threads(8)
  self.net=S().eval();self.net.load_state_dict(torch.load(EXPERT/'vendor/syncnet_v2.model',map_location='cpu',weights_only=True));self.net.requires_grad_(False)
  self.renderer=Renderer();self.mapping=json.loads((OUT/'aux-map.json').read_text());self.audio=np.load(OUT/'audio-features.npy',mmap_mode='r');rig=np.load(WORK/'arkit-student/rig-metrics.npz');self.active=rig['active'];names=json.loads((WORK/'arkit-student/channels.json').read_text());self.mask=torch.tensor([name.startswith(('mouth','jaw')) for name in names]);self.reference={}
 def render(self,pred,index):
  row=self.mapping[index];left=torch.tensor(row['left']);fraction=torch.tensor(row['fraction'],dtype=pred.dtype)[:,None];clipped=pred.clamp(0,1);w=clipped[left]*(1-fraction)+clipped[left+1]*fraction;full=torch.zeros((5,52),dtype=pred.dtype);full[:,self.active]=w;full=full*self.mask
  return self.renderer(full).permute(1,0,2,3)[None]
 def distance(self,pred,index):return torch.linalg.vector_norm(self.net.forward_lip(self.render(pred,index))[0]-torch.tensor(self.audio[index].copy()))
 def gradient(self,pred,target,index):
  if index not in self.reference:
   with torch.no_grad():self.reference[index]=float(self.distance(torch.tensor(target),index))
  p=torch.tensor(pred,requires_grad=True);distance=self.distance(p,index);loss=torch.relu(distance-self.reference[index]);grad=torch.autograd.grad(loss,p)[0]
  if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite expert gradient')
  return float(loss.detach()),grad.numpy(),dict(distance=float(distance.detach()),reference=self.reference[index],active=bool(loss.detach()>0),coefficient_gradient_norm=float(grad.norm()))

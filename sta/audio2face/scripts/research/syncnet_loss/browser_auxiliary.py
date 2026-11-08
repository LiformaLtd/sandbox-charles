"""Training-only numerical SyncNet gradient on the actual Three James renderer."""
from common import *
import numpy as np,torch,cv2,urllib.request,base64,time
from SyncNetModel import S
class BrowserExpert:
 def __init__(self):
  torch.set_num_threads(8);self.net=S().eval();self.net.load_state_dict(torch.load(EXPERT/'vendor/syncnet_v2.model',map_location='cpu',weights_only=True));self.net.requires_grad_(False);self.mapping=json.loads((OUT/'aux-map.json').read_text());self.audio=np.load(OUT/'audio-features.npy',mmap_mode='r');self.active=np.load(WORK/'arkit-student/rig-metrics.npz')['active'];names=json.loads((WORK/'arkit-student/channels.json').read_text());self.mask=np.array([name.startswith(('mouth','jaw')) and i in self.active for i,name in enumerate(names)],np.float32);self.reference={};self.endpoint=json.loads((OUT/'renderer-endpoint.json').read_text())['endpoint']
 def coefficients(self,pred,index):
  row=self.mapping[index];left=torch.tensor(row['left']);fraction=torch.tensor(row['fraction'],dtype=pred.dtype)[:,None];clipped=pred.clamp(0,1);w=clipped[left]*(1-fraction)+clipped[left+1]*fraction;full=torch.zeros((5,52),dtype=pred.dtype);full[:,self.active]=w;return full*torch.tensor(self.mask)
 def images(self,weights):
  data=json.dumps(np.asarray(weights).tolist()).encode();req=urllib.request.Request(self.endpoint,data=data,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=120) as response:images=json.loads(response.read())
  out=[]
  for image in images:
   frame=cv2.imdecode(np.frombuffer(base64.b64decode(image),np.uint8),cv2.IMREAD_COLOR);assert frame.shape==(224,224,3);out.append(cv2.resize(frame[16:176,32:192],(224,224)))
  return np.stack(out)
 def scores(self,clips,index):
  clips=np.asarray(clips,np.float32);images=self.images(clips.reshape(-1,52));video=images.reshape(len(clips),5,224,224,3).transpose(0,4,1,2,3).copy();target=torch.tensor(self.audio[index].copy())
  with torch.inference_mode():
   outputs=[torch.linalg.vector_norm(self.net.forward_lip(torch.tensor(video[i:i+8],dtype=torch.float32))-target,dim=-1).numpy() for i in range(0,len(video),8)]
  return np.concatenate(outputs)
 def estimate(self,weights,index,seed):
  rng=np.random.default_rng(seed);directions=rng.choice(np.array([-1.,1.],np.float32),size=(16,5,52))*self.mask;epsilon=.003;probes=np.concatenate([np.clip(weights+epsilon*directions,0,1),np.clip(weights-epsilon*directions,0,1)]);distances=self.scores(probes,index);gradient=np.mean(((distances[:16]-distances[16:])/(2*epsilon))[:,None,None]*directions,axis=0);return gradient,dict(probe_epsilon=epsilon,directions=16,probe_mean=float(distances.mean()))
 def gradient(self,pred,target,index,seed=37):
  began=time.monotonic();p=torch.tensor(pred,requires_grad=True);weights=self.coefficients(p,index);w=weights.detach().numpy()
  if index not in self.reference:self.reference[index]=float(self.scores(self.coefficients(torch.tensor(target),index).numpy()[None],index)[0])
  distance=float(self.scores(w[None],index)[0]);loss=max(distance-self.reference[index],0);detail=dict(distance=distance,reference=self.reference[index],active=loss>0)
  if loss>0:
   gradient,extra=self.estimate(w,index,seed);grad=torch.autograd.grad((weights*torch.tensor(gradient)).sum(),p)[0].numpy();detail.update(extra)
  else:grad=np.zeros_like(pred)
  assert np.isfinite(grad).all();detail.update(coefficient_gradient_norm=float(np.linalg.norm(grad)),seconds=time.monotonic()-began)
  return loss,grad,detail

"""MLX training implementation references for exported waveform/crop inference."""
from common import *
import numpy as np,mlx.core as mx
from mlx.utils import tree_unflatten
import onnx
from onnx import numpy_helper
from model import Model
from data import Dataset,TRAIN,batch

def main():
 row=json.loads((SCALE/'datasets/old-train/manifest.json').read_text())[0];pcm=np.fromfile(row['audio'],'<f4');n=min(4,len(range(-8000,len(pcm)-4000,8000)));ids=np.full((1,4),-1);ids[0,:n]=np.arange(n);b=batch(Dataset(TRAIN),ids);waves=[]
 for start in range(-8000,-8000+n*8000,8000):
  x=np.zeros(16000,np.float32);lo,hi=max(start,0),min(start+16000,len(pcm));x[lo-start:hi-start]=pcm[lo:hi];waves.append(x)
 values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};result=dict(pcm=np.stack(waves),valid=b['valid'].reshape(4,30)[:n])
 for arm in ['control','recurrent']:
  net=Model(values,arm=='recurrent')
  with np.load(OUT/arm/'best.npz') as z:net.update(tree_unflatten([(k,mx.array(z[k])) for k in z.files]))
  h=mx.zeros((1,32));tracks=[];states=[]
  for t in range(n):
   pred,h=net(mx.array(b['mel'][:,t:t+1]),mx.array(b['valid'][:,t*30:(t+1)*30]),h);full=np.zeros((30,52),np.float32);full[:,np.load(WORK/'arkit-student/rig-metrics.npz')['active']]=np.array(mx.clip(pred[0],0,1));tracks.append(full);states.append(np.array(h))
  result[arm]=np.stack(tracks);result[arm+'_states']=np.stack(states)
 np.savez(OUT/'native-reference.npz',**result)
if __name__=='__main__':main()

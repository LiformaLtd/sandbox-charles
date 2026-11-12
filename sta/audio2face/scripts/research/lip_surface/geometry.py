"""Dense raw-teacher surface loss via exact linear-rig sufficient statistics."""
import numpy as np
from common import *
PAIRS=np.array([[9013,10613],[941,10433],[8859,10434],[2963,3726],[1439,1475],[5980,7505],[6093,18810],[263,18809],[17362,13358]])
def prepare():
 with np.load(ASSETS/'model_data_James.npz') as m,np.load(ASSETS/'bs_skin_James.npz') as r:
  skin=m['neutral_skin'].copy();neutral=r['neutral'].copy();names=[v.decode() for v in r['poseNames'][1:]]
  active=np.flatnonzero(json.loads((ASSETS/'bs_skin_config_James.json').read_text())['blendshape_params']['bsSolveActivePoses'])
  ids=np.flatnonzero((abs(skin[:,0])<3)&(skin[:,1]>166.1)&(skin[:,1]<169)&(skin[:,2]>5))
  assert len(ids)==2267 and set(PAIRS.ravel())<=set(ids.tolist())
  basis=np.stack([r[names[i]][ids] for i in active]).astype(np.float32)
  offset=(skin-neutral)[ids].astype(np.float32)
  gram=np.stack([np.einsum('iv,jv->ij',basis[:,:,axis],basis[:,:,axis],optimize=False)/len(ids) for axis in range(3)]).astype(np.float32)
  return dict(ids=ids,active=active,basis=basis,offset=offset,gram=gram,skin=skin)
def statistics(geometry,rig):
 t=geometry[...,:72006].reshape(*geometry.shape[:-1],24002,3)[...,rig['ids'],:]+rig['offset']
 q=np.einsum('...va,cva->...ac',t,rig['basis'],optimize=False)/len(rig['ids'])
 squared=np.square(t).mean(axis=-2)
 return np.concatenate((q.reshape(*q.shape[:-2],117),squared),axis=-1).astype(np.float32)
def squared_error(weights,stats,gram):
 q=stats[...,:117].reshape(*stats.shape[:-1],3,39);s=stats[...,117:]
 return np.einsum('...i,aij,...j->...a',weights,gram,weights,optimize=False)-2*np.einsum('...ac,...c->...a',q,weights,optimize=False)+s

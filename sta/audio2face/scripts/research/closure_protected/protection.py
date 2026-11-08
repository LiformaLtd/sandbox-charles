"""Project the actual AdamW displacement onto two guard halfspaces."""
from common import *
import numpy as np,mlx.core as mx
from mlx.utils import tree_flatten,tree_unflatten

def pack(tree):return mx.concatenate([v.reshape(-1) for _,v in tree_flatten(tree)])
def unpack(vector,template):
 rows=[];offset=0
 for key,value in tree_flatten(template):
  rows.append((key,vector[offset:offset+value.size].reshape(value.shape)));offset+=value.size
 assert offset==vector.size
 return tree_unflatten(rows)
def dot(a,b):return float(mx.sum(a*b))

def project(delta,guards):
 G=np.array([[dot(a,b) for b in guards] for a in guards],np.float64);b=np.array([dot(g,delta) for g in guards]);norm2=dot(delta,delta);tol=1e-7*np.sqrt(np.maximum(np.diag(G)*norm2,0))+1e-12
 candidates=[]
 if np.all(b<=tol):candidates.append((0.,np.zeros(2)))
 for i in range(2):
  if G[i,i]<=1e-30:continue
  lam=np.zeros(2);lam[i]=b[i]/G[i,i]
  if lam[i]>=0 and np.all(b-G@lam<=tol):candidates.append((float(lam@G@lam),lam))
 lam=np.linalg.pinv(G,rcond=1e-10)@b
 if np.all(lam>=-1e-10) and np.all(b-G@lam<=tol):
  lam=np.maximum(lam,0);candidates.append((float(lam@G@lam),lam))
 if candidates:
  _,lam=min(candidates,key=lambda x:x[0]);result=delta-sum(float(v)*g for v,g in zip(lam,guards))
 else:result=mx.zeros_like(delta)
 after=np.array([dot(g,result) for g in guards]);allowed=5*tol
 if np.any(after>allowed):result=mx.zeros_like(delta);after=np.zeros(2)
 return result,dict(before=b.tolist(),after=after.tolist(),tolerance=allowed.tolist(),projected=bool(np.any(b>tol)),step_norm=float(np.sqrt(norm2)),projected_step_norm=float(np.sqrt(max(0,dot(result,result)))))

"""One fixed training batch sets the additional gradient norm to25% of the main loss."""
from common import *
import numpy as np,torch
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper
import mlx.core as mx,mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_backend import Student,objective
from features import Dataset
from geometry import prepare
from loss import surface_loss

def main():
 train=Dataset(['old-train','new-train']);replay=Dataset(['replay-train','extra-train']);rng=np.random.default_rng(1037);groups=[np.flatnonzero(train.voices==v) for v in sorted(set(train.voices))];indices=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16)
 fresh=train.batch(indices);past=replay.batch(old);batch={k:np.concatenate((fresh[k],past[k])) for k in fresh};parts=[np.load(OUT/'surface-maps'/n/'stats.npy',mmap_mode='r') for n in ['old-train','new-train']];target=np.empty((48,30,120),np.float32);start=0
 for part in parts:
  take=(indices>=start)&(indices<start+len(part));target[take]=part[indices[take]-start];start+=len(part)
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];names=json.loads((WORK/'arkit-student/channels.json').read_text());basis=mx.array(rig['gap_basis'][active]);rest=mx.array(rig['rest']+rig['gap_offset']);lips=mx.array(np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active],np.float32));gram=mx.array(prepare()['gram'])
 net=Student({v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer})
 base,g=nn.value_and_grad(net,objective)(net,*[mx.array(batch[k]) for k in ['mel','weights','gaps','valid']],basis,rest,lips)
 dense,h=nn.value_and_grad(net,surface_loss)(net,mx.array(fresh['mel']),mx.array(target),mx.array(fresh['valid']),gram)
 norm=lambda grads:float(np.sqrt(sum(float(mx.sum(v*v)) for _,v in tree_flatten(grads))))
 a,b=norm(g),norm(h);coefficient=.25*a/b
 ga=dict(tree_flatten(g));hb=dict(tree_flatten(h));cosine=sum(float(mx.sum(ga[k]*hb[k])) for k in ga)/(a*b)
 assert np.isfinite(coefficient) and a>0 and b>0
 freeze(OUT/'calibration.json',dict(indices=indices.tolist(),replay_indices=old.tolist(),main_gradient_norm=a,surface_gradient_norm=b,surface_coefficient=coefficient,ratio=.25,gradient_cosine=cosine,main_loss=float(base),surface_mse_cm2=float(dense),source_sha256=sha(__file__)))
 print(json.dumps(json.loads((OUT/'calibration.json').read_text()),indent=2))
if __name__=='__main__':main()

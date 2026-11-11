"""Prove hint supervision reaches the compact encoder without updating mentor."""
from common import *
import numpy as np,onnx,mlx.core as mx,mlx.nn as nn
from onnx import numpy_helper
from mlx.utils import tree_flatten
from model import Predictor,Learner,training_loss

def main():
 lock();mx.random.seed(37)
 values={t.name:numpy_helper.to_array(t).copy() for t in onnx.load(BASELINE).graph.initializer}
 net=Learner(values);mentor=Predictor(values,192)
 with np.load(OUT/'mentor-s37/best.npz') as z:mentor.load_weights([(k,mx.array(z[k].transpose(0,2,1) if z[k].ndim==3 else z[k])) for k in z.files],strict=False)
 mentor.freeze();before={k:np.array(v).copy() for k,v in tree_flatten(mentor.parameters())}
 data=dataset(['old-train']).batch(np.arange(4));x,y,g,m=[mx.array(data[k]) for k in ['mel','weights','gaps','valid']]
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];basis=mx.array(rig['gap_basis'][active]);rest=mx.array(rig['rest']+rig['gap_offset']);names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=mx.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active])
 hints=[mx.stop_gradient(v[:,15:45]) for v in mentor.hidden(x)]
 grad=nn.value_and_grad(net,training_loss)
 v0,g0=grad(net,x,y,g,m,hints,basis,rest,lips,mx.array(0.))
 v1,g1=grad(net,x,y,g,m,hints,basis,rest,lips,mx.array(.1))
 difference=np.array(g1['student']['encoder']['weight']-g0['student']['encoder']['weight'])
 norm=float(np.linalg.norm(difference));assert norm>1e-6
 for k,v in tree_flatten(mentor.parameters()):np.testing.assert_array_equal(before[k],np.array(v))
 write(OUT/'gradient-flow.json',dict(passed=True,source_sha256=sha(__file__),hint_encoder_gradient_delta_norm=norm,main_loss=float(v0),distillation_loss=float(v1),mentor_unchanged=True))
 print('Hint gradients reach encoder; frozen mentor unchanged. Delta norm:',norm,flush=True)
if __name__=='__main__':main()

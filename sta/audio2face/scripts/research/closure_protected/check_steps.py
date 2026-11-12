"""Real-data guarded-step contract checks before freezing a training run."""
from common import *
import numpy as np,onnx,mlx.core as mx,mlx.nn as nn,mlx.optimizers as optim
from onnx import numpy_helper
from mlx_backend import Student
from features import Dataset
from loss import contact,timing,objective
from protection import pack,unpack
from step import protected_step

def main():
 mx.set_default_device(mx.gpu);values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Student(values);template=net.trainable_parameters();initial=pack(template);mx.eval(initial);train=Dataset(['old-train','new-train']);replay=Dataset(['replay-train','extra-train']);groups=[np.flatnonzero(train.voices==v) for v in sorted(set(train.voices))];rng=np.random.default_rng(1037);rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=mx.array(rig['gap_basis'][rig['active']]);rest=mx.array(rig['rest']+rig['gap_offset']);names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=mx.array(np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in rig['active']],np.float32));functions=[nn.value_and_grad(net,f) for f in [objective,contact,timing]];rows=[]
 for i in range(16):
  net.update(unpack(initial,template));ix=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16);a,b=train.batch(ix),replay.batch(old);batch={k:np.concatenate((a[k],b[k])) for k in a};args=[mx.array(batch[k]) for k in ['mel','weights','gaps','valid']]+[basis,rest,lips];parts=[f(net,*args) for f in functions];grads,_=optim.clip_grad_norm(parts[0][1],1.);optimizer=optim.AdamW(learning_rate=.00012,betas=[.9,.999],eps=1e-8,weight_decay=.015,bias_correction=True);info=protected_step(net,optimizer,grads,[p[1] for p in parts[1:]],[p[0] for p in parts[1:]],args);assert all(v<=limit for v,limit in zip(info['new_guards'],info['guard_limits']));assert info['actual_step_norm']<=info['step_norm']*1.001+1e-7;rows.append(info)
 freeze(OUT/'step-tests.json',dict(rows=rows,passes=True,nonzero_steps=sum(r['actual_step_norm']>1e-10 for r in rows),source_sha256=sha(__file__),step_sha256=sha(HERE/'step.py')));print('Real guarded steps:',sum(r['actual_step_norm']>1e-10 for r in rows),'/16 nonzero; no batch-guard increase beyond tolerance.',flush=True)
if __name__=='__main__':main()

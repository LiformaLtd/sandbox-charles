"""Training-only gradient conflict and optimizer-step checks before training."""
from common import *
import numpy as np,onnx,mlx.core as mx,mlx.nn as nn,mlx.optimizers as optim
from onnx import numpy_helper
from mlx_backend import Student
from features import Dataset
from loss import fitting,contact,timing,objective
from protection import pack,unpack,dot,project

def main():
 assert sha(BASELINE)==EXPECTED;mx.set_default_device(mx.gpu);values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Student(values);train=Dataset(['old-train','new-train']);replay=Dataset(['replay-train','extra-train']);groups=[np.flatnonzero(train.voices==v) for v in sorted(set(train.voices))];rng=np.random.default_rng(1037);rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=mx.array(rig['gap_basis'][rig['active']]);rest=mx.array(rig['rest']+rig['gap_offset']);names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=mx.array(np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in rig['active']],np.float32));rows=[]
 freeze(OUT/'preflight-protocol.json',dict(baseline_sha256=EXPECTED,batches=64,sampler_seed=1037,sampling='same six/voice plus16 replay as trainer; fixed baseline and fresh optimizer each probe',threshold='Proceed if at least10% fitting/contact or fitting/timing gradient conflicts; no test-set input',sources={n:sha(HERE/n) for n in ['common.py','loss.py','protection.py','preflight.py']}))
 functions=[nn.value_and_grad(net,f) for f in [fitting,contact,timing,objective]]
 for step in range(64):
  ix=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16);a,b=train.batch(ix),replay.batch(old);batch={k:np.concatenate((a[k],b[k])) for k in a};args=[mx.array(batch[k]) for k in ['mel','weights','gaps','valid']]+[basis,rest,lips];parts=[f(net,*args) for f in functions];grads=[pack(g) for _,g in parts];f,c,t,total=grads;residual=np.sqrt(dot(total-f-c-t,total-f-c-t)/max(dot(total,total),1e-30));assert residual<1e-5
  params=net.trainable_parameters();before=pack(params);clipped,_=optim.clip_grad_norm(parts[-1][1],1.);optimizer=optim.AdamW(learning_rate=.00012,betas=[.9,.999],eps=1e-8,weight_decay=.015,bias_correction=True);optimizer.update(net,clipped);delta=pack(net.trainable_parameters())-before;projected,info=project(delta,[c,t]);net.update(unpack(before,params));den=lambda x,y:max(np.sqrt(dot(x,x)*dot(y,y)),1e-30);rows.append(dict(batch=step,fit_contact_cosine=dot(f,c)/den(f,c),fit_timing_cosine=dot(f,t)/den(f,t),sum_gradient_relative_error=residual,**info))
  print('batch',step,'contact cosine',round(rows[-1]['fit_contact_cosine'],4),flush=True)
 conflicts={k:float(np.mean([r[k]<-1e-6 for r in rows])) for k in ['fit_contact_cosine','fit_timing_cosine']};freeze(OUT/'conflicts.json',dict(rows=rows,conflict_fractions=conflicts,proceed=bool(max(conflicts.values())>=.1),method='Fixed baseline training-only probes; first-order projection is not a guarantee of held-out contacts.'))
 print(conflicts,flush=True)
if __name__=='__main__':main()

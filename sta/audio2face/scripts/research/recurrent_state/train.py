"""Matched stream sampler, complete backbone fine-tuning and dev-only selection."""
from common import *
import argparse,time,numpy as np
import mlx.core as mx,mlx.nn as nn,mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
import onnx
from onnx import numpy_helper
from data import Dataset,TRAIN,DEV,clips,batch
from model import Model,objective

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['control','recurrent'],required=True);p.add_argument('--benchmark',action='store_true');args=p.parse_args()
 mx.random.seed(37);values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Model(values,args.arm=='recurrent')
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=mx.array(rig['gap_basis'][rig['active']]);rest=mx.array(rig['rest']+rig['gap_offset']);names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=mx.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in rig['active']])
 train=Dataset(TRAIN);dev=Dataset(DEV);devclips=clips(DEV);optimizer=optim.AdamW(.00012,betas=[.9,.999],eps=1e-8,weight_decay=.015,bias_correction=True)
 grad=nn.value_and_grad(net,objective);state=mx.zeros((20,32));schedule=np.load(OUT/'schedule.npz');ix,resets=schedule['indices'],schedule['resets']
 def evaluate():
  accum={}
  for groupstart in range(0,len(devclips),20):
   group=devclips[groupstart:groupstart+20];h=mx.zeros((len(group),32))
   for start in range(0,max(c['count'] for c in group),4):
    ids=np.full((len(group),4),-1)
    for j,c in enumerate(group):
     count=max(0,min(4,c['count']-start));ids[j,:count]=c['start']+np.arange(start,start+count)
    b=batch(dev,ids);pred,h=net(mx.array(b['mel']),mx.array(b['valid']),h);out=np.array(mx.clip(pred,0,1)@basis+rest)
    for j,c in enumerate(group):
     mask=b['valid'][j];actual=b['gaps'][j,mask];estimated=out[j,mask];a=np.median(actual,-1)<=.1;z=np.median(estimated,-1)<=.1
     stat=accum.setdefault(c['voice'],np.zeros(5));stat += [np.square(estimated-actual).sum(),actual.size,(a&z).sum(),a.sum(),z.sum()]
  detail=[dict(voice=v,rms=float(np.sqrt(s/n)),f1=float(2*t/max(1,a+b))) for v,(s,n,t,a,b) in accum.items()]
  return float(np.mean([d['rms']+.25*(1-d['f1']) for d in detail])),detail
 def save(path):np.savez(path,**{n:np.array(v) for n,v in tree_flatten(net.trainable_parameters())})
 run=OUT/args.arm;run.mkdir(exist_ok=True)
 if args.benchmark:
  began=time.monotonic()
  for s in range(3):
   b=batch(train,ix[s]);value,grads=grad(net,*[mx.array(b[k]) for k in ['mel','weights','gaps','valid']],state,basis,rest,lips);mx.eval(value,grads)
  print('BENCHMARK seconds',time.monotonic()-began,'loss',float(value[0]),flush=True);return
 protocol=json.loads((OUT/'protocol.json').read_text())
 for name,digest in protocol['sources'].items():assert sha(APP/name)==digest,name
 assert sha(BASELINE)==EXPECTED and sha(OUT/'schedule.npz')==protocol['schedule_sha256']
 assert not (run/'complete.json').exists()
 best,detail=evaluate();beststep=0;save(run/'best.npz');freeze(run/'initial-dev.json',dict(score=best,voices=detail));began=time.monotonic();history=[];losses=[]
 for s in range(1240):
  state=mx.where(mx.array(resets[s,:,None]),mx.zeros_like(state),mx.stop_gradient(state));b=batch(train,ix[s]);optimizer.learning_rate=float(.000012+.5*(.00012-.000012)*(1+np.cos(np.pi*s/1240)))
  (value,nextstate),grads=grad(net,*[mx.array(b[k]) for k in ['mel','weights','gaps','valid']],state,basis,rest,lips);grads,norm=optim.clip_grad_norm(grads,1.);optimizer.update(net,grads);mx.eval(value,nextstate,norm,net.parameters(),optimizer.state);state=nextstate
  if not np.isfinite(float(value)):raise FloatingPointError('Invalid training loss')
  losses.append(float(value))
  if (s+1)%124==0:
   score,detail=evaluate()
   if score<best:best,beststep=score,s+1;save(run/'best.npz')
   history.append(dict(step=s+1,selection=score,voices=detail,best=best,beststep=beststep,loss=float(np.mean(losses)),seconds=time.monotonic()-began));losses=[]
   (run/'progress.json').write_text(json.dumps(history,indent=2));save(run/'last.npz')
   np.savez(run/'optimizer.npz',**{n:np.array(v) for n,v in tree_flatten(optimizer.state)},stream_state=np.array(state))
   print(args.arm,s+1,'dev',round(score,6),'best',beststep,'seconds',round(time.monotonic()-began),flush=True)
 freeze(run/'complete.json',dict(beststep=beststep,best=best,seconds=time.monotonic()-began,steps=1240,schedule_sha256=sha(OUT/'schedule.npz'),parameters=sum(v.size for _,v in tree_flatten(net.trainable_parameters())),best_sha256=sha(run/'best.npz')))
if __name__=='__main__':main()

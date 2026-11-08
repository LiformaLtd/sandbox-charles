"""Full-batch diagnostic of current v4 learnability, not held-out quality."""
from common import *
import argparse,time,numpy as np
import mlx.core as mx,mlx.nn as nn,mlx.optimizers as optim
from mlx.utils import tree_flatten
import onnx
from onnx import numpy_helper
from mlx_backend import Student,objective

def loss(net,mel,weights,gaps,valid,basis,rest,lips,fit_only):
 if not fit_only:return objective(net,mel,weights,gaps,valid,basis,rest,lips)
 pred=net(mel)[:,15:45];return (mx.square((pred-weights)/net.scale)*valid[...,None]).sum()/(valid.sum()*39)

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['mixed','fit-mse'],required=True);args=p.parse_args();protocol=json.loads((OUT/'protocol.json').read_text())
 for name,digest in protocol['source_sha256'].items():assert sha(APP/name)==digest,name
 assert sha(BASELINE)==EXPECTED and sha(OUT/'batch.npz')==protocol['batch_sha256'];mx.random.seed(37)
 values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Student(values);run=OUT/args.arm;run.mkdir(exist_ok=True);assert not (run/'result.json').exists()
 with np.load(OUT/'batch.npz') as z:b={k:z[k] for k in ['mel','weights','gaps','valid']}
 tensors=[mx.array(b[k]) for k in ['mel','weights','gaps','valid']];rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=mx.array(rig['gap_basis'][rig['active']]);rest=mx.array(rig['rest']+rig['gap_offset']);names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=mx.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in rig['active']]);fit=b['weights']@np.array(basis)+np.array(rest);mask=b['valid'];truth=b['gaps'];metadata=json.loads((OUT/'manifest.json').read_text())
 def evaluate():
  pred=np.array(mx.clip(net(tensors[0])[:,15:45],0,1));g=pred@np.array(basis)+np.array(rest);a=np.median(truth[mask],-1)<=.1;c=np.median(g[mask],-1)<=.1
  result=dict(student_to_fit_rms_mm=float(10*np.sqrt(np.square(g[mask]-fit[mask]).mean())),student_to_raw_rms_mm=float(10*np.sqrt(np.square(g[mask]-truth[mask]).mean())),fitting_rms_mm=float(10*np.sqrt(np.square(fit[mask]-truth[mask]).mean())),normalized_coefficient_mae=float(np.abs((pred-b['weights'])/np.array(net.scale))[mask].mean()),contact_f1=float(2*(a&c).sum()/max(1,a.sum()+c.sum())))
  detail=[];cursor=0
  for row in metadata:
   n=row['windows'];m=mask[cursor:cursor+n];x=g[cursor:cursor+n];y=truth[cursor:cursor+n];f=fit[cursor:cursor+n];detail.append(dict(stem=row['stem'],voice=row['engine']+':'+row['voice'],frames=int(m.sum()),raw_rms_mm=float(10*np.sqrt(np.square(x[m]-y[m]).mean())),fit_rms_mm=float(10*np.sqrt(np.square(x[m]-f[m]).mean())),reference_rms_mm=float(10*np.sqrt(np.square(f[m]-y[m]).mean()))));cursor+=n
  result['macro_raw_rms_mm']=float(np.mean([d['raw_rms_mm'] for d in detail]));result['macro_reference_rms_mm']=float(np.mean([d['reference_rms_mm'] for d in detail]));result['clips']=detail
  return result
 def save(path):np.savez(path,**net.torch_weights())
 initial=evaluate();best=initial;beststep=0;save(run/'best.npz');history=[];optimizer=optim.AdamW(.0003,weight_decay=0.);gradient=nn.value_and_grad(net,loss);began=time.monotonic()
 for step in range(1,3001):
  optimizer.learning_rate=float(.00001+.5*(.0003-.00001)*(1+np.cos(np.pi*(step-1)/3000)))
  value,grads=gradient(net,*tensors,basis,rest,lips,args.arm=='fit-mse');grads,norm=optim.clip_grad_norm(grads,1.);optimizer.update(net,grads);mx.eval(value,norm,net.parameters(),optimizer.state)
  if not np.isfinite(float(value)):raise FloatingPointError('Invalid loss')
  if step==1 or step%100==0:
   current=evaluate();history.append(dict(step=step,loss=float(value),**current))
   if current['student_to_fit_rms_mm']<best['student_to_fit_rms_mm']:best=current;beststep=step;save(run/'best.npz')
   (run/'progress.json').write_text(json.dumps(dict(initial=initial,best=best,beststep=beststep,history=history),indent=2));save(run/'last.npz');np.savez(run/'optimizer.npz',**{n:np.array(v) for n,v in tree_flatten(optimizer.state)})
   print(args.arm,step,'fit',round(current['student_to_fit_rms_mm'],5),'raw',round(current['student_to_raw_rms_mm'],5),'reference',round(current['fitting_rms_mm'],5),'seconds',round(time.monotonic()-began),flush=True)
 result=dict(arm=args.arm,initial=initial,best=best,beststep=beststep,final=current,history=history,steps=3000,seconds=time.monotonic()-began,best_sha256=sha(run/'best.npz'),training_only=True,close_to_subset_reference=best['student_to_raw_rms_mm']<=1.25*best['fitting_rms_mm'],parameters=sum(v.size for _,v in tree_flatten(net.trainable_parameters())))
 freeze(run/'result.json',result)
if __name__=='__main__':main()

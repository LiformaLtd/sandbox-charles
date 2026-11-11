"""Matched frozen-vs-learned mel banks; predictor and losses remain the same."""
from common import *
import argparse,importlib.util,time
import numpy as np,onnx
from onnx import numpy_helper
import mlx.core as mx,mlx.nn as nn,mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
from model import Model,loss
from data import Dataset

def snapshot(net):return {k:np.array(v) for k,v in tree_flatten(net.trainable_parameters())}
def evaluate(net,datasets,basis,rest):
 scores=[]
 for data in datasets:
  accum={}
  for start in range(0,data.length,128):
   ix=np.arange(start,min(start+128,data.length));batch=data.batch(ix)
   predicted=np.array(mx.clip(net(mx.array(batch['power']))[:,15:45],0,1)@basis+rest)
   for voice in sorted(set(data.voices[ix])):
    take=data.voices[ix]==voice;valid=batch['valid'][take];a=batch['gaps'][take][valid];b=predicted[take][valid]
    actual,estimated=np.median(a,-1)<=.1,np.median(b,-1)<=.1
    stat=accum.setdefault(str(voice),np.zeros(5,np.float64));stat += [np.square(a-b).sum(),a.size,(actual&estimated).sum(),actual.sum(),estimated.sum()]
  for voice,(sq,n,h,a,b) in accum.items():scores.append(dict(voice=voice,rms=float(np.sqrt(sq/n)),f1=float(2*h/max(1,a+b))))
 return dict(score=float(np.mean([x['rms']+.25*(1-x['f1']) for x in scores])),rms=float(np.mean([x['rms'] for x in scores])),f1=float(np.mean([x['f1'] for x in scores])),voices=scores)

def export(net,values,active,run,learn):
 import torch
 torch.set_num_threads(2)
 spec=importlib.util.spec_from_file_location('frontend_export',APP/'scripts/research/compact_ablation/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 student=base.Student(96,values['student.mean'],values['student.scale'],active)
 state={k:torch.from_numpy(values['student.'+k].copy()) for k in student.state_dict()}
 for k,v in net.student.torch_weights().items():state[k]=torch.from_numpy(v)
 student.load_state_dict(state);student.eval();model=base.WaveformModel(student).eval()
 bank=np.array(net.frontend.bank());assert np.isfinite(bank).all() and (bank>=0).all()
 with torch.no_grad():model.mel.copy_(torch.from_numpy(bank[:,:,None]))
 path=run/'model.onnx'
 torch.onnx.export(model,torch.zeros(1,16000),str(path),input_names=['pcm'],output_names=['blendshapes'],opset_version=17,dynamo=False)
 graph=onnx.load(path);onnx.checker.check_model(graph);original=onnx.load(BASELINE)
 ops=lambda g:[(n.op_type,list(n.input),list(n.output),[a.SerializeToString() for a in n.attribute]) for n in g.graph.node]
 if ops(graph)!=ops(original):raise ValueError('Export topology changed')
 tensors={t.name:numpy_helper.to_array(t) for t in graph.graph.initializer}
 for key in ['filters','student.mean','student.scale','student.left','student.fraction','student.channel_map']:
  np.testing.assert_array_equal(tensors[key],values[key])
 if not learn:np.testing.assert_array_equal(bank,values['mel'][:,:,0])
 np.testing.assert_array_equal(tensors['mel'][:,:,0],bank)
 if path.stat().st_size!=2251827:raise ValueError('Artifact size changed')
 return dict(sha256=sha(path),bytes=path.stat().st_size,predictor_parameters=331047,additional_learned_frontend_parameters=16448 if learn else 0,bank_sha256=__import__('hashlib').sha256(bank.tobytes()).hexdigest(),bank_delta_rms=float(np.sqrt(np.mean((bank-values['mel'][:,:,0])**2))),bank_positive_entries=int((bank>0).sum()),base_positive_entries=int((values['mel']>0).sum()))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['fixed','learned'],required=True);parser.add_argument('--seed',type=int,default=37);args=parser.parse_args()
 p=lock();learn=args.arm=='learned'
 if args.seed not in p['seeds']:raise ValueError('Unregistered seed')
 run=OUT/f'{args.arm}-s{args.seed}';run.mkdir(parents=True,exist_ok=True)
 if (run/'progress.json').exists() or (run/'complete.json').exists():raise ValueError('Existing trajectory; never overwrite')
 mx.set_default_device(mx.gpu);mx.random.seed(args.seed);rng=np.random.default_rng(args.seed+1000)
 values={t.name:numpy_helper.to_array(t).copy() for t in onnx.load(BASELINE).graph.initializer};net=Model(values,learn)
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];basis=mx.array(rig['gap_basis'][active]);rest=mx.array(rig['rest']+rig['gap_offset'])
 names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=mx.array(np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active],np.float32))
 train=Dataset(['old-train','new-train']);replay=Dataset(['replay-train','extra-train']);dev=[Dataset(['common-dev']),Dataset(['replay-dev'])]
 groups=[np.flatnonzero(train.voices==v) for v in sorted(set(train.voices))]
 if len(groups)!=8:raise ValueError('Voice split changed')
 options=dict(learning_rate=.00012,betas=[.9,.999],eps=1e-8,weight_decay=.015,bias_correction=True)
 predictor_optimizer=optim.AdamW(**options);frontend_optimizer=optim.AdamW(**{**options,'weight_decay':0.})
 gradient=nn.value_and_grad(net,loss);initial=evaluate(net,dev,basis,rest)
 initial_sha=__import__('hashlib').sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(snapshot(net).items()))).hexdigest()
 best=initial['score'];best_step=0;history=[];losses=[];sampler=__import__('hashlib').sha256()
 np.savez(run/'best.npz',**snapshot(net));write(run/'initial.json',dict(metrics=initial,parameters_sha256=initial_sha,protocol_sha256=sha(OUT/'protocol.json')))
 begin=time.monotonic()
 for step in range(1,p['steps']+1):
  indices=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16);sampler.update(indices.tobytes()+old.tobytes())
  a,b=train.batch(indices),replay.batch(old);batch={k:np.concatenate([a[k],b[k]]) for k in a}
  rate=float(.000012+.5*(.00012-.000012)*(1+np.cos(np.pi*(step-1)/p['steps'])))
  predictor_optimizer.learning_rate=rate;frontend_optimizer.learning_rate=rate*.01
  value,grads=gradient(net,*[mx.array(batch[k]) for k in ['power','weights','gaps','valid']],basis,rest,lips)
  sg,snorm=optim.clip_grad_norm(grads['student'],1.);fg,fnorm=optim.clip_grad_norm(grads['frontend'],1.)
  predictor_optimizer.update(net.student,sg);frontend_optimizer.update(net.frontend,fg);net.frontend.project()
  mx.eval(value,snorm,fnorm,net.parameters(),predictor_optimizer.state,frontend_optimizer.state)
  if not all(np.isfinite(float(v)) for v in [value,snorm,fnorm]):raise FloatingPointError('Nonfinite update')
  losses.append(float(value))
  if step%124==0:
   current=evaluate(net,dev,basis,rest)
   if current['score']<best:
    best,best_step=current['score'],step;np.savez(run/'best.npz',**snapshot(net))
   history.append(dict(step=step,**current,loss=float(np.mean(losses)),seconds=time.monotonic()-begin,frontend_gradient_norm=float(fnorm),student_gradient_norm=float(snorm)));losses=[]
   (run/'progress.json').write_text(json.dumps(dict(initial=initial,best=best,best_step=best_step,history=history),indent=2))
   if step%744==0:
    arrays={'model.'+k:v for k,v in snapshot(net).items()}
    for name,opt in [('student_optimizer',predictor_optimizer),('frontend_optimizer',frontend_optimizer)]:arrays.update({name+'.'+k:np.array(v) for k,v in tree_flatten(opt.state)})
    np.savez(run/'checkpoint.part.npz',**arrays);(run/'checkpoint.part.npz').replace(run/'checkpoint.npz')
    (run/'checkpoint.json').write_text(json.dumps(dict(step=step,rng=rng.bit_generator.state,protocol_sha256=sha(OUT/'protocol.json')),indent=2))
   print(args.arm,args.seed,step,'score',round(current['score'],6),'best',best_step,'sec',round(time.monotonic()-begin),flush=True)
 with np.load(run/'best.npz') as z:net.update(tree_unflatten([(k,mx.array(z[k])) for k in z.files]))
 selected=evaluate(net,dev,basis,rest)
 report=dict(arm=args.arm,seed=args.seed,steps=p['steps'],initial=initial,selected=selected,best_step=best_step,sampler_sha256=sampler.hexdigest(),initial_parameters_sha256=initial_sha,protocol_sha256=sha(OUT/'protocol.json'),seconds=time.monotonic()-begin,export=export(net,values,active,run,learn))
 write(run/'complete.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()

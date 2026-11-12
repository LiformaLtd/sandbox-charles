"""Train an independent-channel mentor or matched compact control/distillation."""
from common import *
import argparse, importlib.util, math, time
import numpy as np, onnx
from onnx import numpy_helper
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
from model import Predictor,Learner,training_loss
from mlx_backend import objective

def evaluate(net, datasets, basis, rest):
    scores=[]
    for data in datasets:
        accum={}
        for start in range(0,data.length,128):
            ix=np.arange(start,min(start+128,data.length)); batch=data.batch(ix)
            prediction=np.array(mx.clip(net(mx.array(batch['mel']))[:,15:45],0,1)@basis+rest)
            for voice in sorted(set(data.voices[ix])):
                take=data.voices[ix]==voice; mask=batch['valid'][take]
                truth=batch['gaps'][take][mask]; pred=prediction[take][mask]
                a,b=np.median(truth,-1)<=.1,np.median(pred,-1)<=.1
                stat=accum.setdefault(str(voice),np.zeros(5,np.float64))
                stat += [np.square(truth-pred).sum(),truth.size,(a&b).sum(),a.sum(),b.sum()]
        for voice,(sq,n,h,a,b) in accum.items(): scores.append(dict(voice=voice,rms=float(np.sqrt(sq/n)),f1=float(2*h/max(1,a+b))))
    return dict(score=float(np.mean([s['rms']+.25*(1-s['f1']) for s in scores])),rms=float(np.mean([s['rms'] for s in scores])),f1=float(np.mean([s['f1'] for s in scores])),voices=scores)

def export(net, values, active, width, run):
    import torch
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('intermediate_export',APP/'scripts/research/compact_ablation/model.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    student=base.Student(width,values['student.mean'],values['student.scale'],active)
    state=student.state_dict()
    for k in state:
        if 'student.'+k in values and tuple(values['student.'+k].shape)==tuple(state[k].shape): state[k]=torch.from_numpy(values['student.'+k].copy())
    for k,v in net.torch_weights().items(): state[k]=torch.from_numpy(v)
    student.load_state_dict(state); student.eval()
    path=run/'model.onnx'
    torch.onnx.export(base.WaveformModel(student).eval(),torch.zeros(1,16000),str(path),input_names=['pcm'],output_names=['blendshapes'],opset_version=17,dynamo=False)
    graph=onnx.load(path);onnx.checker.check_model(graph)
    if any('head' in x.name for x in graph.graph.initializer): raise ValueError('Training projection leaked')
    if width==96:
        original=onnx.load(BASELINE)
        ops=lambda g:[(n.op_type,list(n.input),list(n.output),[a.SerializeToString() for a in n.attribute]) for n in g.graph.node]
        if ops(original)!=ops(graph): raise ValueError('Compact graph changed')
        if sum(v.numel() for v in student.parameters())!=331047: raise ValueError('Compact capacity changed')
    return dict(sha256=sha(path),bytes=path.stat().st_size,parameters=sum(v.numel() for v in student.parameters()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['mentor','control','distill'],required=True);p.add_argument('--seed',type=int,default=37);args=p.parse_args()
    protocol=lock(); seed=args.seed
    if seed not in protocol['students']['seeds']: raise ValueError('Unregistered seed')
    if args.arm=='mentor' and seed!=37: raise ValueError('Unregistered mentor')
    run=OUT/(args.arm+'-s'+str(seed));run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists() or (run/'checkpoint.json').exists(): raise ValueError('Existing completed/interrupted trajectory; never overwrite')
    mx.set_default_device(mx.gpu);mx.random.seed(seed);rng=np.random.default_rng(seed+1000)
    values={t.name:numpy_helper.to_array(t).copy() for t in onnx.load(BASELINE).graph.initializer}
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];basis=mx.array(rig['gap_basis'][active]);rest=mx.array(rig['rest']+rig['gap_offset'])
    names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=mx.array(np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1 for i in active],np.float32))
    width=192 if args.arm=='mentor' else 96
    net=Predictor(values,width) if args.arm=='mentor' else Learner(values)
    student=net if args.arm=='mentor' else net.student
    initial_hash=__import__('hashlib').sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(student.torch_weights().items()))).hexdigest()
    mentor=None
    if args.arm!='mentor':
        info=json.loads((OUT/'mentor-s37/complete.json').read_text())
        if not info['mentor_gate_pass']: raise ValueError('Mentor failed development screen; stop')
        mentor=Predictor(values,192)
        with np.load(OUT/'mentor-s37/best.npz') as z:
            mentor.load_weights([(k,mx.array(z[k].transpose(0,2,1) if z[k].ndim==3 else z[k])) for k in z.files],strict=False)
        mentor.freeze();mx.eval(mentor.parameters())
    train=dataset(['old-train','new-train']);replay=dataset(['replay-train','extra-train']);devs=[dataset(['common-dev']),dataset(['replay-dev'])]
    groups=[np.flatnonzero(train.voices==v) for v in sorted(set(train.voices))]
    if len(groups)!=8: raise ValueError('Training voice split changed')
    steps=protocol['mentor']['steps'] if args.arm=='mentor' else protocol['students']['steps']
    initial=evaluate(student,devs,basis,rest);best=initial['score'];best_step=0;history=[]
    np.savez(run/'best.npz',**student.torch_weights())
    optimizer=optim.AdamW(learning_rate=.00012,betas=[.9,.999],eps=1e-8,weight_decay=.015,bias_correction=True)
    loss_fn=objective if args.arm=='mentor' else training_loss
    gradient=nn.value_and_grad(net,loss_fn)
    def checkpoint(step):
        arrays={'model.'+k:np.array(v) for k,v in tree_flatten(net.trainable_parameters())}
        arrays.update({'optimizer.'+k:np.array(v) for k,v in tree_flatten(optimizer.state)})
        np.savez(run/'checkpoint.part.npz',**arrays);(run/'checkpoint.part.npz').replace(run/'checkpoint.npz')
        meta=dict(step=step,best=best,best_step=best_step,rng=rng.bit_generator.state,history=history,protocol_sha256=sha(OUT/'protocol.json'))
        (run/'checkpoint.json').write_text(json.dumps(meta,indent=2))
    # Fail rather than silently start over after an interrupted run.
    if (run/'checkpoint.json').exists(): raise ValueError('Interrupted run exists; use resume implementation or a fresh output directory')
    begin=time.monotonic();sampler=__import__('hashlib').sha256();losses=[]
    write(run/'initial.json',dict(metrics=initial,student_parameters_sha256=initial_hash,protocol_sha256=sha(OUT/'protocol.json')))
    for step in range(1,steps+1):
        ix=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16);sampler.update(ix.tobytes()+old.tobytes())
        fresh,previous=train.batch(ix),replay.batch(old);batch={k:np.concatenate([fresh[k],previous[k]]) for k in fresh}
        x,y,g,m=[mx.array(batch[k]) for k in ['mel','weights','gaps','valid']]
        optimizer.learning_rate=float(.000012+.5*(.00012-.000012)*(1+np.cos(np.pi*(step-1)/steps)))
        if mentor is None: value,grads=gradient(net,x,y,g,m,basis,rest,lips)
        else:
            hints=[mx.stop_gradient(h[:,15:45]) for h in mentor.hidden(x)];mx.eval(hints)
            coefficient=.1*min(1,step/372) if args.arm=='distill' else 0.
            value,grads=gradient(net,x,y,g,m,hints,basis,rest,lips,mx.array(coefficient))
        grads,norm=optim.clip_grad_norm(grads,1.)
        optimizer.update(net,grads);mx.eval(value,norm,net.parameters(),optimizer.state)
        if not np.isfinite(float(value)) or not np.isfinite(float(norm)): raise FloatingPointError('Nonfinite update')
        losses.append(float(value))
        if step%124==0:
            current=evaluate(student,devs,basis,rest)
            if current['score']<best:
                best,best_step=current['score'],step;np.savez(run/'best.npz',**student.torch_weights())
            history.append(dict(step=step,**current,loss=float(np.mean(losses)),seconds=time.monotonic()-begin));losses=[]
            (run/'progress.json').write_text(json.dumps(dict(initial=initial,best=best,best_step=best_step,history=history),indent=2))
            if step%744==0: checkpoint(step)
            print(args.arm,seed,step,'score',round(current['score'],6),'best',best_step,'sec',round(time.monotonic()-begin),flush=True)
    # Stored weights are in PyTorch convolution layout; load explicitly into MLX.
    selected=np.load(run/'best.npz');student.load_weights([(k,mx.array(selected[k].transpose(0,2,1) if selected[k].ndim==3 else selected[k])) for k in selected.files],strict=False)
    metrics=evaluate(student,devs,basis,rest)
    report=dict(arm=args.arm,seed=seed,steps=steps,initial=initial,selected=metrics,best_step=best_step,sampler_sha256=sampler.hexdigest(),initial_parameters_sha256=initial_hash,protocol_sha256=sha(OUT/'protocol.json'),seconds=time.monotonic()-begin)
    if mentor is None: report['mentor_gate_pass']=bool(metrics['rms']<=.99*initial['rms'] and metrics['f1']>=initial['f1']-.005)
    report['export']=export(student,values,active,width,run)
    write(run/'complete.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()

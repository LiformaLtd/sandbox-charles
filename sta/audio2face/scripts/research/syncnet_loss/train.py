"""Unchanged student; matched training-only frozen SyncNet auxiliary loss."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time
import numpy as np
from common import *
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
os.environ['MLX_ENABLE_TF32']='0'
import mlx.core as mx,mlx.nn as nn,mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
import onnx
from onnx import numpy_helper
from features import Dataset
from mlx_backend import Student
from mlx_backend import objective as original_objective
if os.environ.get('A2F_TORCH_SITE_PACKAGES'):sys.path.append(os.environ['A2F_TORCH_SITE_PACKAGES'])
from browser_auxiliary import BrowserExpert

def objective(net,mel,target,gaps,valid,basis,rest,lips,cotangent,position,strength):
    base=original_objective(net,mel,target,gaps,valid,basis,rest,lips)
    extra=(net(mel[position:position+1])[:,15:45]*cotangent).sum()
    return base+strength*extra


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['control','syncnet'],required=True);p.add_argument('--seed',type=int,default=37);args=p.parse_args()
    protocol=json.loads((OUT/'training-protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        if sha(APP/name)!=digest:raise ValueError('Frozen training source changed')
    run=OUT/'trajectories'/args.arm;run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():raise ValueError('Completed trajectory exists')
    if args.seed not in protocol['allowed_seeds']:raise ValueError('Unregistered training seed')
    mx.set_default_device(mx.gpu);mx.random.seed(args.seed);rng=np.random.default_rng(args.seed+1000)
    values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(protocol['baseline']).graph.initializer}
    net=Student(values);train=Dataset(['old-train','new-train']);dev=Dataset(['common-dev']);replay=Dataset(['replay-train','extra-train']);olddev=Dataset(['replay-dev'])
    voices=sorted(set(train.voices));groups=[np.flatnonzero(train.voices==v) for v in voices]
    if len(groups)!=8:raise ValueError('Expected eight training voices')
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];basis=mx.array(rig['gap_basis'][active]);rest=mx.array(rig['rest']+rig['gap_offset'])
    names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=mx.array(np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active],np.float32))
    optimizer=optim.AdamW(learning_rate=.00012,betas=[.9,.999],eps=1e-8,weight_decay=.015,bias_correction=True)
    gradient=nn.value_and_grad(net,objective)
    auxiliary=BrowserExpert() if args.arm=='syncnet' else None
    auxiliary_map=json.loads((OUT/'aux-map.json').read_text());auxiliary_history=[]
    strength=float(protocol['auxiliary_lambda'])
    def tensors(data):return [mx.array(data[k]) for k in ['mel','weights','gaps','valid']]
    def evaluate():
        scores=[]
        for dataset in [dev,olddev]:
            accum={}
            for start in range(0,dataset.length,128):
                ix=np.arange(start,min(start+128,dataset.length));batch=dataset.batch(ix)
                prediction=mx.clip(net(mx.array(batch['mel']))[:,15:45],0,1)@basis+rest
                prediction=np.array(prediction);valid=batch['valid'];truth=batch['gaps']
                for voice in sorted(set(dataset.voices[ix])):
                    take=dataset.voices[ix]==voice;mask=valid[take];a=truth[take][mask];b=prediction[take][mask]
                    actual=np.median(a,-1)<=.1;estimated=np.median(b,-1)<=.1
                    stat=accum.setdefault(str(voice),np.zeros(5,np.float64));stat += [np.square(a-b).sum(),a.size,(actual&estimated).sum(),actual.sum(),estimated.sum()]
            for voice,(squared,count,hits,a,b) in accum.items():scores.append(dict(voice=voice,rms=float(np.sqrt(squared/count)),f1=float(2*hits/max(1,a+b))))
        return float(np.mean([s['rms']+.25*(1-s['f1']) for s in scores])),scores
    def checkpoint(step,best,best_step,history):
        model={"model."+n:np.array(v) for n,v in tree_flatten(net.trainable_parameters())}
        state={"optimizer."+n:np.array(v) for n,v in tree_flatten(optimizer.state)}
        with np.load(run/'best.npz') as z: best_arrays={'best.'+k:z[k].copy() for k in z.files}
        filename=f'checkpoint-{step:06}.npz'
        temp=run/(filename+'.part.npz');np.savez(temp,**model,**state,**best_arrays);temp.replace(run/filename)
        freeze(run/'recipe.json',dict(arm=args.arm,seed=args.seed,training_protocol_sha256=sha(OUT/'training-protocol.json')))
        meta=dict(step=step,best=best,best_step=best_step,rng=rng.bit_generator.state,history=history,checkpoint_file=filename,checkpoint_sha256=sha(run/filename))
        freeze(run/f'checkpoint-{step:06}.json',meta)
        temp=run/'checkpoint.part.json';temp.write_text(json.dumps(meta,indent=2));temp.replace(run/'checkpoint.json')
    def export(name,best_step,best_score,selected=None):
        # Only parameter values change; the browser graph and normalization stay fixed.
        if os.environ.get('A2F_TORCH_SITE_PACKAGES'):sys.path.append(os.environ['A2F_TORCH_SITE_PACKAGES'])
        import torch
        torch.set_num_threads(2)
        spec=importlib.util.spec_from_file_location('scale_export',OUT/'sources/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
        student=base.Student(96,values['student.mean'],values['student.scale'],active)
        state={k:torch.from_numpy(values['student.'+k].copy()) for k in student.state_dict()}
        if selected is None:
            with np.load(run/'best.npz') as z:selected={k:z[k].copy() for k in z.files}
        for key,value in selected.items():state[key]=torch.from_numpy(value.copy())
        student.load_state_dict(state);student.eval();dest=OUT/name;dest.mkdir(exist_ok=True)
        if (dest/'selection.json').exists():
            previous=json.loads((dest/'selection.json').read_text())
            if sha(dest/'model.onnx')!=previous['model_sha256'] or previous['best_step']!=best_step:raise ValueError('Changed locked export')
            return
        torch.onnx.export(base.WaveformModel(student).eval(),torch.zeros(1,16000),str(dest/'model.part.onnx'),input_names=['pcm'],output_names=['blendshapes'],opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(dest/'model.part.onnx'));(dest/'model.part.onnx').replace(dest/'model.onnx')
        report=dict(arm=args.arm,seed=args.seed,budget=3720,best_step=best_step,best_selection=best_score,model_sha256=sha(dest/'model.onnx'),bytes=(dest/'model.onnx').stat().st_size,parameters=sum(v.numel() for v in student.parameters()),protocol_sha256=sha(OUT/'training-protocol.json'))
        if report['bytes']!=2251827 or report['parameters']!=331047:raise ValueError('Runtime capacity changed')
        freeze(dest/'selection.json',report)
    score,initial=evaluate();best,best_step=score,0;history=[];start_step=0
    meta=run/'checkpoint.json'
    if meta.exists():
        saved=json.loads(meta.read_text());rng.bit_generator.state=saved['rng'];start_step=saved['step'];best,best_step,history=saved['best'],saved['best_step'],saved['history']
        if sha(run/saved['checkpoint_file'])!=saved['checkpoint_sha256']:raise ValueError('Changed checkpoint')
        with np.load(run/saved['checkpoint_file']) as z:
            net.update(tree_unflatten([(k.removeprefix('model.'),mx.array(z[k])) for k in z.files if k.startswith('model.')]))
            optimizer.state=tree_unflatten([(k.removeprefix('optimizer.'),mx.array(z[k])) for k in z.files if k.startswith('optimizer.')])
            np.savez(run/'best.npz',**{k.removeprefix('best.'):z[k].copy() for k in z.files if k.startswith('best.')})
    else:np.savez(run/'best.npz',**net.torch_weights())
    freeze(run/'initial-dev.json',dict(score=score,voices=initial))
    if (run/'auxiliary-history.json').exists():auxiliary_history=json.loads((run/'auxiliary-history.json').read_text())
    began=time.monotonic();losses=[];norms=[];sampler=__import__('hashlib').sha256()
    audit_rng=np.random.default_rng(args.seed+1000)
    for _ in range(start_step):
        prior=np.concatenate([audit_rng.choice(g,6,replace=True) for g in groups]);old=audit_rng.integers(replay.length,size=16);sampler.update(prior.tobytes()+old.tobytes())
    if start_step and audit_rng.bit_generator.state!=rng.bit_generator.state:raise ValueError('Sampler resume changed')
    for step in range(start_step+1,3721):
        indices=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16)
        sampler.update(indices.tobytes()+old.tobytes())
        fresh=train.batch(indices);previous=replay.batch(old);batch={k:np.concatenate((fresh[k],previous[k])) for k in fresh}
        optimizer.learning_rate=float(.000012+.5*(.00012-.000012)*(1+np.cos(np.pi*(step-1)/3720)))
        position=0;cotangent=np.zeros((1,30,39),np.float32);aux_value=0.;weight=0.
        if step%4==0:
            voice=(step//4-1)%8;positions=[j for j in range(voice*6,voice*6+6) if auxiliary_map[int(indices[j])]['eligible']]
            if positions:
                position=positions[0];index=int(indices[position])
                if auxiliary is not None:
                    prediction=np.array(net(mx.array(batch['mel'][position:position+1]))[:,15:45])[0]
                    aux_value,cot,detail=auxiliary.gradient(prediction,batch['weights'][position],index,seed=37000+step);cotangent[0]=cot;weight=strength*min(1.,step/372)
                    auxiliary_history.append(dict(step=step,index=index,loss=aux_value,weight=weight,**detail))
        value,grads=gradient(net,*tensors(batch),basis,rest,lips,mx.array(cotangent),position,weight);grads,norm=optim.clip_grad_norm(grads,1.);optimizer.update(net,grads);mx.eval(value,norm,net.parameters(),optimizer.state)
        if not np.isfinite(float(value)) or not np.isfinite(float(norm)):raise FloatingPointError('Invalid optimizer update')
        reported=float(value)
        if weight:reported += weight*(aux_value-float(np.sum(prediction*cot)))
        losses.append(reported);norms.append(float(norm))
        if step%124==0:
            score,scores=evaluate()
            if score<best:
                best,best_step=score,step;np.savez(run/'best.part.npz',**net.torch_weights());(run/'best.part.npz').replace(run/'best.npz')
            history.append(dict(step=step,selection=score,dev=scores,loss=float(np.mean(losses)),gradient_norm_mean=float(np.mean(norms)),clipped_fraction=float(np.mean(np.array(norms)>1)),elapsed=time.monotonic()-began));losses=[];norms=[]
            checkpoint(step,best,best_step,history)
            (run/'auxiliary-history.json').write_text(json.dumps(auxiliary_history,indent=2))
            (run/'progress.json').write_text(json.dumps(dict(initial=initial,best=best,best_step=best_step,history=history),indent=2))
            print(args.arm,step,'selection',round(score,6),'best',best_step,'seconds',round(time.monotonic()-began),flush=True)
    export(args.arm,best_step,best)
    freeze(run/'complete.json',dict(steps=3720,best_step=best_step,best=best,source_sha256=sha(__file__),sampler_sha256=sampler.hexdigest(),seconds=time.monotonic()-began))


if __name__=='__main__':main()

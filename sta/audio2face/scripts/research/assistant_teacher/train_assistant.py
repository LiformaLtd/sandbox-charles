"""Frozen assistant pilot followed by matched unchanged-size student arms."""
from common import *
import argparse,time,importlib.util
import numpy as np,onnx
from onnx import numpy_helper
import mlx.core as mx,mlx.nn as nn,mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_backend import Student, objective, smooth
from features import Dataset
from model import Assistant,features,load_selected,export

def data():
    train=Dataset(['old-train','new-train']);replay=Dataset(['replay-train','extra-train'])
    groups=[np.flatnonzero(train.voices==v) for v in sorted(set(train.voices))]
    return train,replay,groups,[Dataset(['common-dev']),Dataset(['replay-dev'])]

def evaluate(net,datasets,basis,rest):
    stats={}
    for part,dataset in enumerate(datasets):
        for start in range(0,dataset.length,128):
            ix=np.arange(start,min(start+128,dataset.length));b=dataset.batch(ix);pred=np.array(mx.clip(net(mx.array(b['mel']))[:,15:45],0,1)@basis+rest)
            for voice in sorted(set(dataset.voices[ix])):
                take=dataset.voices[ix]==voice;valid=b['valid'][take];truth=b['gaps'][take][valid];estimate=pred[take][valid];a=np.median(truth,-1)<=.1;c=np.median(estimate,-1)<=.1
                s=stats.setdefault(str(part)+':'+voice,np.zeros(5));s += [np.square(truth-estimate).sum(),truth.size,(a&c).sum(),a.sum(),c.sum()]
    voices={k:dict(rms=float(np.sqrt(s/n)),f1=float(2*hits/max(1,a+b))) for k,(s,n,hits,a,b) in stats.items()}
    rms=float(np.mean([v['rms'] for v in voices.values()]));f1=float(np.mean([v['f1'] for v in voices.values()]))
    return dict(selection=rms+.25*(1-f1),rms=rms,f1=f1,voices=voices)

def batch_sample(train,replay,groups,rng):
    ix=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16);a,b=train.batch(ix),replay.batch(old)
    return {k:np.concatenate([a[k],b[k]]) for k in a},ix.tobytes()+old.tobytes()

def constants():
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];names=json.loads((WORK/'arkit-student/channels.json').read_text())
    return active,mx.array(rig['gap_basis'][active]),mx.array(rig['rest']+rig['gap_offset']),mx.array(np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active],np.float32))

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['assistant','control','distilled'],required=True);args=p.parse_args();run=OUT/args.arm;run.mkdir(exist_ok=True)
    if (run/'selection.json').exists():raise ValueError('Completed arm exists')
    protocol=json.loads((OUT/'training-protocol.json').read_text())
    for name,digest in protocol['sources'].items():
        if sha(APP/name)!=digest:raise ValueError('Frozen source changed: '+name)
    assert sha(BASELINE)==protocol['baseline_sha256']
    mx.set_default_device(mx.gpu);mx.random.seed(37);rng=np.random.default_rng(1037)
    values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Assistant(values) if args.arm=='assistant' else Student(values)
    train,replay,groups,dev=data();active,basis,rest,lips=constants();assistant=None;feature_scale=None
    if args.arm!='assistant':
        selected=json.loads((OUT/'assistant/selection.json').read_text());initial=json.loads((OUT/'preflight.json').read_text())['initial_dev'];best=selected['development']
        gate=best['rms']<=.97*initial['rms'] and best['f1']>=initial['f1']-1e-6
        freeze(OUT/'assistant-gate.json',dict(passes=gate,initial=initial,assistant=best,improvement=1-best['rms']/initial['rms']))
        if not gate:raise ValueError('Assistant quality gate failed; no tiny-student campaign')
    if args.arm=='distilled':
        assert sha(OUT/'assistant/model.onnx')==selected['sha256']
        assistant=Assistant(values);load_selected(assistant,OUT/'assistant/best.npz');assistant.freeze()
        scale_rng=np.random.default_rng(9037);squares=np.zeros(96,np.float64);count=0
        for _ in range(32):
            b,_=batch_sample(train,replay,groups,scale_rng);h=np.array(assistant.feature(mx.array(b['mel']))[:,15:45]);valid=b['valid'];squares+=np.square(h[valid]).sum(0);count+=int(valid.sum())
        feature_scale=mx.array(np.maximum(np.sqrt(squares/count),.1).astype(np.float32));np.save(run/'feature-scale.npy',np.array(feature_scale))
    def loss(model,mel,target,gaps,valid,tp,tf):
        value=objective(model,mel,target,gaps,valid,basis,rest,lips)
        if args.arm=='distilled':
            h=features(model,mel)[:,15:45];count=mx.maximum(valid.sum(),1)
            feature=(((h-tf)/feature_scale)**2*valid[...,None]).sum()/(count*96)
            prediction=model(mel)[:,15:45];kd=(smooth((prediction-tp)/model.scale,.5)*valid[...,None]).sum()/(count*39)
            value=value+.1*feature+.25*kd
        return value
    grad=nn.value_and_grad(net,loss);budget=7440 if args.arm=='assistant' else 3720
    optimizer=optim.AdamW(learning_rate=.00012,betas=[.9,.999],eps=1e-8,weight_decay=.015,bias_correction=True)
    best=evaluate(net,dev,basis,rest);best_step=0;np.savez(run/'best.npz',**net.torch_weights());history=[];sampler=hashlib.sha256();began=time.monotonic();losses=[]
    freeze(run/'recipe.json',dict(arm=args.arm,budget=budget,seed=37,protocol_sha256=sha(OUT/'training-protocol.json'),assistant_sha256=selected['sha256'] if args.arm=='distilled' else None))
    for step in range(1,budget+1):
        b,raw=batch_sample(train,replay,groups,rng);sampler.update(raw);mel,target,gaps,valid=[mx.array(b[k]) for k in ['mel','weights','gaps','valid']]
        if assistant is not None:tf=mx.stop_gradient(assistant.feature(mel)[:,15:45]);tp=mx.stop_gradient(assistant.output(tf)*assistant.scale+assistant.mean)
        else:tp=tf=mx.zeros(1)
        optimizer.learning_rate=float(.000012+.5*(.00012-.000012)*(1+np.cos(np.pi*(step-1)/budget)))
        value,grads=grad(net,mel,target,gaps,valid,tp,tf);grads,norm=optim.clip_grad_norm(grads,1.);optimizer.update(net,grads);mx.eval(value,norm,net.parameters(),optimizer.state)
        if not np.isfinite(float(value)):raise FloatingPointError('Nonfinite training')
        losses.append(float(value))
        if step%248==0:
            score=evaluate(net,dev,basis,rest)
            if score['selection']<best['selection']:best,best_step=score,step;np.savez(run/'best.npz',**net.torch_weights())
            history.append(dict(step=step,development=score,loss=float(np.mean(losses)),seconds=time.monotonic()-began));losses=[]
            (run/'progress.json').write_text(json.dumps(dict(best_step=best_step,best=best,history=history),indent=2));print(args.arm,step,'rms',score['rms'],'f1',score['f1'],'best',best_step,'seconds',round(time.monotonic()-began),flush=True)
    load_selected(net,run/'best.npz');report=export(net,values,active,run/'model.onnx');freeze(run/'selection.json',dict(**report,development=best,best_step=best_step,history=history,sampler_sha256=sampler.hexdigest(),seconds=time.monotonic()-began));print(json.dumps({k:v for k,v in report.items()}),flush=True)

if __name__=='__main__':main()

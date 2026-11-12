"""Matched correction/direct heads; no updates to the frozen v4 model."""
from common import *
import argparse,time,numpy as np,onnx
from onnx import numpy_helper
import mlx.core as mx,mlx.nn as nn,mlx.optimizers as optim
from cache_features import FrozenDataset
from residual_model import Head,export
from mlx_backend import objective

def datasets():
    train=FrozenDataset(['old-train','new-train']);replay=FrozenDataset(['replay-train','extra-train']);dev=[FrozenDataset(['common-dev']),FrozenDataset(['replay-dev'])]
    return train,replay,[np.flatnonzero(train.voices==v) for v in sorted(set(train.voices))],dev
def sample(train,replay,groups,rng):
    ix=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16);a,b=train.batch(ix),replay.batch(old)
    return {k:np.concatenate([a[k],b[k]]) for k in a},ix.tobytes()+old.tobytes()
def constants():
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];names=json.loads((WORK/'arkit-student/channels.json').read_text())
    return active,mx.array(rig['gap_basis'][active]),mx.array(rig['rest']+rig['gap_offset']),mx.array(np.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active],np.float32))
def evaluate(net,datasets,basis,rest):
    stats={}
    for part,dataset in enumerate(datasets):
        for start in range(0,dataset.length,128):
            ix=np.arange(start,min(start+128,dataset.length));b=dataset.batch(ix);pred=np.array(mx.clip(net(mx.array(b['mel']))[:,15:45],0,1)@basis+rest)
            for voice in sorted(set(dataset.voices[ix])):
                take=dataset.voices[ix]==voice;valid=b['valid'][take];truth=b['gaps'][take][valid];estimate=pred[take][valid];a=np.median(truth,-1)<=.1;c=np.median(estimate,-1)<=.1
                s=stats.setdefault(str(part)+':'+voice,np.zeros(5));s += [np.square(truth-estimate).sum(),truth.size,(a&c).sum(),a.sum(),c.sum()]
    voices={k:dict(rms=float(np.sqrt(s/n)),f1=float(2*hits/max(1,a+b))) for k,(s,n,hits,a,b) in stats.items()};rms=float(np.mean([v['rms'] for v in voices.values()]));f1=float(np.mean([v['f1'] for v in voices.values()]))
    return dict(selection=rms+.25*(1-f1),rms=rms,f1=f1,voices=voices)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['control','residual'],required=True);args=p.parse_args();run=OUT/args.arm;run.mkdir(exist_ok=True)
    if (run/'recipe.json').exists():raise ValueError('Trajectory already exists; archive before a clean restart')
    protocol=json.loads((OUT/'training-protocol.json').read_text())
    for name,digest in protocol['sources'].items():assert sha(APP/name)==digest,name
    assert sha(BASELINE)==protocol['baseline_sha256'];values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};head=Head(values,np.load(OUT/'feature-scale.npy'),args.arm)
    mx.set_default_device(mx.gpu);mx.random.seed(37);rng=np.random.default_rng(1037);train,replay,groups,dev=datasets();active,basis,rest,lips=constants();gradient=nn.value_and_grad(head,objective)
    optimizer=optim.AdamW(learning_rate=.0003,betas=[.9,.999],eps=1e-8,weight_decay=.015,bias_correction=True);best=evaluate(head,dev,basis,rest);best_step=0;head.save(run/'best.npz');history=[];began=time.monotonic();sampler=hashlib.sha256();losses=[];norms=[]
    freeze(run/'recipe.json',dict(arm=args.arm,protocol_sha256=sha(OUT/'training-protocol.json'),budget=7440,seed=37,initial_dev=best))
    for step in range(1,7441):
        b,raw=sample(train,replay,groups,rng);sampler.update(raw);optimizer.learning_rate=float(.00003+.5*(.0003-.00003)*(1+np.cos(np.pi*(step-1)/7440)))
        value,grads=gradient(head,*[mx.array(b[k]) for k in ['mel','weights','gaps','valid']],basis,rest,lips);grads,norm=optim.clip_grad_norm(grads,1.);optimizer.update(head,grads);mx.eval(value,norm,head.parameters(),optimizer.state)
        if not np.isfinite(float(value)):raise FloatingPointError('Nonfinite head update')
        losses.append(float(value));norms.append(float(norm))
        if step%248==0:
            score=evaluate(head,dev,basis,rest)
            if score['selection']<best['selection']:best,best_step=score,step;head.save(run/'best.npz')
            history.append(dict(step=step,development=score,loss=float(np.mean(losses)),gradient_norm=float(np.mean(norms)),clipped_fraction=float(np.mean(np.array(norms)>1)),seconds=time.monotonic()-began));losses=[];norms=[]
            (run/'progress.json').write_text(json.dumps(dict(best_step=best_step,best=best,history=history),indent=2));print(args.arm,step,'rms',round(score['rms'],6),'f1',round(score['f1'],6),'best',best_step,flush=True)
    head.restore(run/'best.npz');report=export(head,values,active,run/'model.onnx');freeze(run/'selection.json',dict(**report,development=best,best_step=best_step,sampler_sha256=sampler.hexdigest(),seconds=time.monotonic()-began,history=history));print(json.dumps(report),flush=True)

if __name__=='__main__':main()

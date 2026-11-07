"""One-factor compact-v3 ablations. Historical corpus manifests are input artifacts."""
from pathlib import Path
import os
import argparse, importlib.util, json, hashlib, math, time, copy, sys
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
APP=Path(__file__).resolve().parents[3]
HERE=Path(os.environ.get('A2F_ABLATION_OUTPUT',APP/'.local/compact-ablation')).resolve()
HERE.mkdir(parents=True,exist_ok=True)
WORK=Path(os.environ.get('A2F_RESEARCH_WORK',APP/'.local/research-corpora')).resolve()
import onnx
from onnx import numpy_helper
import model as base
V3=WORK/'arkit-student-v3'
BASELINE=V3/'james-arkit-student-v3.onnx'
ROOTS=[V3,V3/'extra',WORK/'arkit-student',WORK/'arkit-student-v2']

@lru_cache(maxsize=16)
def input_fingerprint(root,split):
    root=Path(root)
    digest=hashlib.sha256((root/'manifest.json').read_bytes())
    digest.update(str(root).encode())
    for row in json.loads((root/'manifest.json').read_text()):
        if row['split']==split:
            digest.update((root/'labels'/f'{row["stem"]}.npz').read_bytes())
            digest.update(Path(row['audio']).read_bytes())
    return digest.hexdigest()

def dataset(root,split,active,context):
    return _dataset(str(root),split,tuple(int(i) for i in active),context)

@lru_cache(maxsize=16)
def _dataset(root,split,active,context):
    root=Path(root)
    active=np.asarray(active,dtype=np.int64)
    cache=HERE/'features';cache.mkdir(exist_ok=True)
    key=input_fingerprint(str(root),split)[:16]
    path=cache/f'{key}-{split}-{context}.npz'
    if path.exists():
        with np.load(path) as z:return tuple(z[str(i)] for i in range(4))
    xs,ys,gs,ms=[],[],[],[]
    padding=(context-16000)//2
    offsets=np.arange(15,45)*16000//60
    for row in json.loads((root/'manifest.json').read_text()):
        if row['split']!=split:continue
        with np.load(root/'labels'/f'{row["stem"]}.npz') as z:weights,gaps=z['weights'][:,active],z['gaps']
        pcm=np.fromfile(row['audio'],'<f4');cursor=0
        for original_start in range(-8000,len(pcm)-4000,8000):
            start=original_start-padding
            x=np.zeros(context,np.float32);lo,hi=max(start,0),min(start+context,len(pcm))
            x[lo-start:hi-start]=pcm[lo:hi]
            valid=(original_start+offsets>=0)&(original_start+offsets<len(pcm));count=int(valid.sum())
            y=np.zeros((30,len(active)),np.float32);g=np.zeros((30,9),np.float32)
            y[valid]=weights[cursor:cursor+count];g[valid]=gaps[cursor:cursor+count]
            xs.append(x);ys.append(y);gs.append(g);ms.append(valid);cursor+=count
        assert cursor==len(weights)
    def features_of(pcm):
        windows=np.lib.stride_tricks.sliding_window_view(np.pad(pcm,((0,0),(200,200))),400,axis=1)[:,::160]
        fft=np.fft.rfft(windows*base.HANN,n=512,axis=-1)
        power=(fft.real**2+fft.imag**2).astype(np.float32)
        mel=np.einsum('btf,mf->btm',power,base.MEL,optimize=False)
        result=np.log1p(mel*10).transpose(0,2,1).copy()
        assert np.isfinite(result).all()
        return result
    features=np.concatenate([features_of(np.stack(xs[i:i+128])) for i in range(0,len(xs),128)])
    result=features,np.stack(ys),np.stack(gs),np.stack(ms)
    temporary=path.with_name(path.stem+f'.{os.getpid()}.npz')
    np.savez(temporary,**{str(i):v for i,v in enumerate(result)})
    temporary.replace(path)
    return result

def initialize(width,context,dilated=False):
    graph=onnx.load(BASELINE)
    values={t.name:numpy_helper.to_array(t).copy() for t in graph.graph.initializer}
    active=np.load(WORK/'arkit-student/rig-metrics.npz')['active']
    net=base.Student(width,values['student.mean'],values['student.scale'],active)
    indices=np.arange(width)%96
    counts=np.bincount(indices,minlength=96)
    state=net.state_dict()
    for key in state:
        old=values['student.'+key]
        if key=='encoder.weight':old=old[indices]
        elif key=='encoder.bias':old=old[indices]
        elif key.startswith('blocks.'):
            if key.endswith('weight'):old=old[indices][:,indices]/counts[indices][None,:,None]
            else:old=old[indices]
        elif key=='output.weight':old=old[:,indices]/counts[indices][None,:,None]
        state[key]=torch.from_numpy(old.astype(state[key].numpy().dtype))
    net.load_state_dict(state)
    if dilated:
        net.blocks[5].conv.dilation=(32,)
        net.blocks[5].conv.padding=(32,)
    if context!=16000:
        positions=(np.arange(context*60//16000)*16000//60).astype(np.float32)/160
        net.left=torch.from_numpy(positions.astype(np.int64))
        net.fraction=torch.from_numpy(positions%1).view(1,1,-1)
    # Noise breaks duplicated-channel symmetry after function-preserving widening.
    if width!=96:
        with torch.no_grad():
            for name,p in net.named_parameters():
                if 'weight' in name:p.add_(torch.randn_like(p)*1e-5)
    return net,active

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--width',type=int,default=96)
    p.add_argument('--context',type=int,default=16000);p.add_argument('--extra',action='store_true');p.add_argument('--dilated-context',action='store_true')
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--lr',type=float,default=.00006);p.add_argument('--seed',type=int,default=37)
    args=p.parse_args()
    recipe=hashlib.sha256(Path(__file__).read_bytes()+Path(base.__file__).read_bytes()).hexdigest()
    baseline_sha=hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    inputs={f'{root}:{split}':input_fingerprint(str(root),split)
            for root in ROOTS for split in ('train','dev')
            if split=='train' or root!=ROOTS[1]}
    if args.extra:inputs['additional_train']=input_fingerprint(str(HERE/'extra'),'train')
    input_sha=hashlib.sha256(json.dumps({'baseline':baseline_sha,'datasets':inputs},sort_keys=True).encode()).hexdigest()
    complete=HERE/(args.name+'.json')
    if complete.exists() and (HERE/(args.name+'.onnx')).exists():
        previous=json.loads(complete.read_text())
        previous.get('config',{}).setdefault('dilated_context',False)
        if 'bytes' in previous and previous.get('config')==vars(args) and previous.get('recipe_sha256')==recipe and previous.get('input_sha256')==input_sha:
            print('Reusing completed identical run',args.name,flush=True);return
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    # Capture the unchanged control's RNG state separately. Width must not
    # change training-window permutations by consuming more initializer draws.
    initialize(96,16000)
    data_rng=torch.Generator();data_rng.set_state(torch.get_rng_state())
    torch.manual_seed(args.seed)
    net,active=initialize(args.width,args.context,args.dilated_context)
    tensor=torch.as_tensor
    cat=lambda roots,split:tuple(tensor(np.concatenate([dataset(r,split,active,args.context)[i] for r in roots])) for i in range(4))
    # Use a fixed optimizer-step budget for all ablations, even with added data.
    original_new=cat(ROOTS[:2],'train');steps=math.ceil(len(original_new[0])/48)
    new=cat(ROOTS[:2]+([HERE/'extra'] if args.extra else []),'train') if args.extra else original_new
    old=cat(ROOTS[2:],'train')
    dev=[cat([r],'dev') for r in (V3,ROOTS[3],ROOTS[2])]
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=tensor(rig['gap_basis'][active]);rest=tensor(rig['rest']+rig['gap_offset'])
    names=json.loads((WORK/'arkit-student/channels.json').read_text())
    lips=tensor(np.asarray([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active],np.float32))
    start=15+(args.context-16000)*60//32000
    def prediction(x):return net(x)[:,start:start+30]
    def loss(pred,target,gaps,valid):
        w=valid[...,None];count=valid.sum().clamp(min=1)
        shape=(F.smooth_l1_loss((pred-target)/net.scale,torch.zeros_like(pred),reduction='none',beta=.5)*w*lips).sum()/(count*len(active))
        geom=pred.clamp(0,1)@basis+rest
        gap=(F.smooth_l1_loss(geom,gaps,reduction='none',beta=.14)*w).sum()/(count*9)
        adjacent=valid[:,1:]&valid[:,:-1]
        velocity=(((pred.diff(dim=1)-target.diff(dim=1)).abs()/net.scale)*adjacent[...,None]).sum()/(count*len(active))
        actual=gaps.median(dim=-1).values;estimate=geom.median(dim=-1).values;contact=actual<=.1
        contact_loss=(F.binary_cross_entropy_with_logits((.1-estimate)/.16,contact.float(),reduction='none')*(1+contact)*valid).sum()/count
        near=((actual[:,1:]<.35)|(actual[:,:-1]<.35))&adjacent
        timing=(F.smooth_l1_loss(estimate.diff(dim=1),actual.diff(dim=1),reduction='none',beta=.1)*near).sum()/near.sum().clamp(min=1)
        return shape+1.2*gap+.12*velocity+.22*contact_loss+.45*timing
    @torch.no_grad()
    def evaluate():
        net.eval();scores=[]
        for x,y,g,m in dev:
            pred=torch.cat([prediction(v).clamp(0,1) for v in x.split(64)])
            pg=(pred@basis+rest)[m];truth=g[m]
            a=truth.median(-1).values<=.1;b=pg.median(-1).values<=.1
            scores.append([float(2*(a&b).sum()/(a.sum()+b.sum()).clamp(min=1)),float((pg-truth).square().mean().sqrt())])
        return scores
    def selection(scores):
        (f0,e0),(f1,e1),(f2,e2)=scores
        return f0+.6*f1+.25*f2-.3*e0-.2*e1-.1*e2
    initial=evaluate();best=selection(initial);best_state=copy.deepcopy(net.state_dict());best_epoch=0
    log={'recipe_sha256':recipe,'input_sha256':input_sha,'dataset_fingerprints':inputs,'initial_model_sha256':baseline_sha,'config':vars(args),'initial_dev':initial,'history':[],'optimizer_steps_per_epoch':steps,'new_windows':len(new[0]),'old_windows':len(old[0])}
    assert all(torch.isfinite(v).all() for ds in [new,old,*dev] for v in ds)
    print(args.name,'initial',initial,'steps',steps,flush=True)
    optim=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.015)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optim,steps*args.epochs,eta_min=args.lr*.1)
    begun=time.monotonic()
    for epoch in range(args.epochs):
        net.train()
        reference_order=torch.randperm(len(original_new[0]),generator=data_rng)
        replay=torch.randperm(len(old[0]),generator=data_rng)
        order=(torch.randperm(len(new[0]),generator=torch.Generator().manual_seed(args.seed+1000+epoch))
               if args.extra else reference_order)
        for i in range(steps):
            index=order[torch.arange(i*48,i*48+48)%len(order)];oi=replay[i*16:i*16+16]
            x,y,g,m=(torch.cat((n[index],o[oi])) for n,o in zip(new,old))
            optim.zero_grad(set_to_none=True);objective=loss(prediction(x),y,g,m)
            if not torch.isfinite(objective): raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}, step {i}')
            objective.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optim.step();schedule.step()
        scores=evaluate();score=selection(scores)
        log['history'].append({'epoch':epoch+1,'dev':scores,'selection':score,'elapsed':time.monotonic()-begun})
        if score>best:best=score;best_state=copy.deepcopy(net.state_dict());best_epoch=epoch+1
        print(args.name,epoch+1,'dev',scores,'best',best_epoch,'elapsed',round(time.monotonic()-begun),flush=True)
        (HERE/f'{args.name}.json').write_text(json.dumps(log,indent=2))
        torch.save({'epoch':epoch+1,'model':net.state_dict(),'best_model':best_state,
                    'best_epoch':best_epoch,'best_selection':best,'optimizer':optim.state_dict(),
                    'scheduler':schedule.state_dict(),'data_rng':data_rng.get_state(),
                    'torch_rng':torch.get_rng_state(),'config':vars(args),'recipe_sha256':recipe},
                   HERE/f'{args.name}-training-state.pt')
    net.load_state_dict(best_state);net.eval();torch.save(net.state_dict(),HERE/f'{args.name}.pt')
    model=base.WaveformModel(net).eval();probe=torch.zeros(1,args.context)
    torch.onnx.export(model,probe,str(HERE/f'{args.name}.onnx'),input_names=['pcm'],output_names=['blendshapes'],opset_version=17,dynamo=False)
    graph=onnx.load(HERE/f'{args.name}.onnx');onnx.checker.check_model(graph)
    log.update(best_epoch=best_epoch,best_dev=evaluate(),parameters=sum(x.numel() for x in net.parameters()),bytes=(HERE/f'{args.name}.onnx').stat().st_size,seconds=time.monotonic()-begun)
    (HERE/f'{args.name}.json').write_text(json.dumps(log,indent=2))
if __name__=='__main__':main()

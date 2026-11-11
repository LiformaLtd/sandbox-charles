"""Optimize the delivered constrained corner motion, retaining frozen expression heads."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import *
from guard_mouth_tails import GapGuardFace
from eyelid_model import LIDS
from full_neutral_model import conditions
from fit import PAIRS

DEST=APP/'.local/mouth-tail-gap-guard-v3'
CURRENT=APP/'.local/mouth-tail-gap-guard-v2'
DATA=APP/'.local/expressive-full-neutral-v1'
STEPS=600


def main():
    torch.set_num_threads(4);torch.manual_seed(79);rng=np.random.default_rng(2079)
    DEST.mkdir(exist_ok=True)
    if (DEST/'selection.json').exists():raise ValueError('Completed projected refinement exists')
    ds={}
    for split in ['train','dev']:
        with np.load(DATA/(split+'.npz')) as z:ds[split]={k:z[k].copy() for k in ['hidden','baseline','target','gaps','valid','emotion','voices','cases']}
        d=ds[split];n=len(d['hidden']);following=np.arange(n);tail=np.zeros_like(d['valid'])
        totals={case:int(d['valid'][d['cases']==case].sum()) for case in set(d['cases'])};cursor={c:0 for c in totals}
        for i,case in enumerate(d['cases']):
            valid=d['valid'][i];count=int(valid.sum());tail[i,valid]=np.arange(cursor[case],cursor[case]+count)>=totals[case]-48;cursor[case]+=count
            if i+11<n and case==d['cases'][i+11]:following[i]=i+11
        overlap=np.concatenate([d['valid'][:,25:30],d['valid'][following,:5]],axis=1);overlap[following==np.arange(n)]=False
        d['following']=following;d['overlap']=overlap;d['tail']=tail
    net=GapGuardFace();net.load_state_dict(torch.load(CURRENT/'face.pt',map_location='cpu',weights_only=True));net.requires_grad_(False)
    # Frozen features and temporal convolutions; change only the fast output
    # projection. Brow/lid/gaze heads and the compact articulation prior stay fixed.
    net.fast.output.requires_grad_(True)
    with np.load(OUT/'calibration.npz') as z:
        corner=torch.from_numpy(z['shape'][:,PAIRS[[0,-1]]].mean(2).reshape(52,6));gap=torch.from_numpy(z['gap_basis'])
    for d in ds.values():d['corner_target']=np.einsum('ntc,ck->ntk',d['target'],corner.numpy(),optimize=False)
    stress_path=APP/'.local/corner-appointment-regression/features.npz'
    with np.load(stress_path) as z:ds['regression']={k:z[k].copy() for k in z.files}
    def tensors(d,ix):
        h=torch.from_numpy(d['hidden'][ix]);b=torch.from_numpy(d['baseline'][ix]);e=torch.from_numpy(d['emotion'][ix,None,:]).expand(-1,60,-1)
        style=torch.zeros(len(h),60,3);style[:,:,1]=1
        return h,b,e,style
    def masked(x,v):return (x.square()*v[:,:,None]).sum()/(v.sum().clamp_min(1)*x.shape[-1])
    # Cache the two references once. No waveform-model training or changes.
    for split,d in ds.items():
        parent=[];previous=[]
        with torch.inference_mode():
            for start in range(0,len(d['hidden']),32):
                ix=np.arange(start,min(start+32,len(d['hidden'])));h,b,e,s=tensors(d,ix)
                parent.append(net.parent(h,b,e,s).numpy());previous.append(net(h,b,e,s).numpy())
        d['parent']=np.concatenate(parent);d['previous']=np.concatenate(previous)
        print('cached',split,flush=True)
    train=ds['train'];voices=sorted(set(train['voices']))
    groups=[np.flatnonzero(train['voices']==v) for v in voices]
    angry=[np.flatnonzero((train['voices']==v)&(train['emotion'][:,EMOTIONS.index('anger')]>.9)) for v in voices]
    active=[c for c in range(52) if c not in [*LIDS,1,2,3,4,8,9,10,11,41,42,43,44,45]]
    scales=[]
    for e in range(11):
        ix=np.arange(e,len(train['target']),11);y=train['target'][ix][train['valid'][ix]];scales.append(np.maximum(y.std(0),.03))
    scale_table=torch.from_numpy(np.stack(scales))
    def loss_for(d,ix,p,e):
        p=p[:,15:45];y=torch.from_numpy(d['target'][ix]);v=torch.from_numpy(d['valid'][ix]);tail=torch.from_numpy(d['tail'][ix])
        old=torch.from_numpy(d['previous'][ix,15:45]);parent=torch.from_numpy(d['parent'][ix,15:45]);sigma=conditions(e[:,15:45])@scale_table
        truth=torch.from_numpy(d['corner_target'][ix])
        delta=p@corner-truth;old_delta=old@corner-truth;parent_delta=parent@corner-truth
        fit=masked(delta/.1,v)+2*masked(delta/.1,v&tail)
        velocity=masked((delta[:,1:]-delta[:,:-1])/.02,v[:,1:]&v[:,:-1])
        reference=torch.minimum(old_delta.square().mean(-1),parent_delta.square().mean(-1))
        regression=(torch.relu(delta.square().mean(-1)-reference)*v*(1+2*tail)).sum()/(v.sum().clamp_min(1)*.1**2)
        anchor=masked((p[:,:,active]-old[:,:,active])/sigma[:,:,active],v)
        lip=masked(((p-parent)@gap)/.005,v)
        return 8*fit+2*velocity+8*regression+.1*anchor+2*lip
    @torch.inference_mode()
    def evaluate():
        d=ds['dev'];loss=0.;n=0;errors=np.zeros(11);counts=np.zeros(11);before=np.zeros(11);parents=np.zeros(11)
        for start in range(0,len(d['hidden']),32):
            ix=np.arange(start,min(start+32,len(d['hidden'])));h,b,e,s=tensors(d,ix);p=net(h,b,e,s)
            nh,nb,ne,ns=tensors(d,d['following'][ix]);following=net(nh,nb,ne,ns)
            consistency=masked(((p[:,40:50]-following[:,10:20])@corner)/.02,torch.from_numpy(d['overlap'][ix]))
            loss+=float(loss_for(d,ix,p,e)+8*consistency)*len(ix);n+=len(ix)
            target=torch.from_numpy(d['target'][ix]);delta=((p[:,15:45]-target)@corner).numpy()
            old=((torch.from_numpy(d['previous'][ix,15:45])-target)@corner).numpy();parent=((torch.from_numpy(d['parent'][ix,15:45])-target)@corner).numpy()
            for j,k in enumerate(ix):
                condition=0 if d['emotion'][k].sum()==0 else int(d['emotion'][k].argmax())+1;valid=d['valid'][k]
                errors[condition]+=np.square(delta[j,valid]).sum();before[condition]+=np.square(old[j,valid]).sum();parents[condition]+=np.square(parent[j,valid]).sum();counts[condition]+=delta[j,valid].size
        ratios=np.sqrt(errors/np.maximum(before,1e-12));prior=np.sqrt(errors/np.maximum(parents,1e-12))
        # Prioritize the observed neutral/anger scope, while checking all controls.
        eligible=all(ratios[i]<=1.05 and prior[i]<=1.05 for i in [0,2]) and ratios.max()<=1.1
        stress=ds['regression'];ix=np.arange(len(stress['hidden']));h,b,e,s=tensors(stress,ix)
        sp=net(h,b,e,s)[:,15:45]@corner;v=stress['valid'];values=sp.numpy()[v];truth=stress['corner_target'][v]
        stress_rms=float(np.sqrt(np.mean((values[-48:]-truth[-48:])**2)))*10
        eligible=eligible and stress_rms<=.9115
        return loss/n,ratios.tolist(),prior.tolist(),eligible,stress_rms
    freeze(DEST/'protocol.json',dict(steps=STEPS,seed=79,source_sha256=sha(__file__),parent_sha256=sha(CURRENT/'model.onnx'),
        data_sha256={n:sha(stress_path if n=='regression' else DATA/(n+'.npz')) for n in ds},trained='Only fast output projection; gradient passes through the 128 bounded geometry projection iterations',
        sampling='One anger and one random-condition window per training voice, plus explicit appointment/anger raw-geometry development windows; not independent validation',
        losses='Constrained teacher corner fit/velocity, per-frame non-regression versus both references, overlap continuity, prior coefficient anchor, residual lip-gap penalty',
        selection='Held-out voice dev objective; neutral/anger within 5% of both references, other conditions within 10%; minimum 200 steps',
        direct_gate='Strong-anger appointment tail error must return to <= pre-refinement 0.9115 mm; retain neutral gains and lip-gap RMS'))
    optimizer=torch.optim.AdamW(net.fast.output.parameters(),lr=.0002,weight_decay=.001)
    best=float('inf');best_state=None;history=[];begun=time.monotonic()
    for step in range(1,STEPS+1):
        ix=np.concatenate([[rng.choice(a),rng.choice(g)] for a,g in zip(angry,groups)]);h,b,e,s=tensors(train,ix);p=net(h,b,e,s)
        nh,nb,ne,ns=tensors(train,train['following'][ix]);following=net(nh,nb,ne,ns)
        continuity=masked(((p[:,40:50]-following[:,10:20])@corner)/.02,torch.from_numpy(train['overlap'][ix]))
        stress=ds['regression'];last=np.arange(max(0,len(stress['hidden'])-3),len(stress['hidden']))
        si=np.concatenate([rng.choice(last,3,replace=True),rng.integers(len(stress['hidden']),size=1)])
        sh,sb,se,ss=tensors(stress,si);sp=net(sh,sb,se,ss)
        loss=loss_for(train,ix,p,e)+8*continuity+.5*loss_for(stress,si,sp,se)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite projected loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.fast.output.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        optimizer.param_groups[0]['lr']=float(.0002*(.1+.45*(1+np.cos(np.pi*step/STEPS))))
        if step%100==0:
            score,ratios,prior,eligible,stress_rms=evaluate();eligible=eligible and step>=200
            if eligible and score<best:best=score;best_step=step;best_state=copy.deepcopy(net.state_dict());torch.save(best_state,DEST/'best.pt')
            history.append(dict(step=step,score=score,condition_ratios=ratios,parent_condition_ratios=prior,regression_corner_rms_mm=stress_rms,eligible=eligible,seconds=time.monotonic()-begun))
            (DEST/'progress.json').write_text(json.dumps(history,indent=2));print('projected',step,round(score,3),'neutral/anger ratios',round(ratios[0],3),round(ratios[2],3),'regression mm',round(stress_rms,3),'eligible',eligible,'seconds',round(time.monotonic()-begun),flush=True)
    if best_state is None:raise ValueError('No projected checkpoint met held-out non-regression gates')
    torch.save(best_state,DEST/'face.pt');freeze(DEST/'selection.json',dict(best_step=best_step,score=best,history=history));print('SELECTED',best_step,flush=True)

if __name__=='__main__':main()

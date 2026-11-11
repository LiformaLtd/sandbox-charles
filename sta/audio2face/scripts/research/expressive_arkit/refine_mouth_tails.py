"""Fit teacher corner motion and window continuity, guarded by raw lip-gap RMS."""
import copy
import json
import time
import numpy as np
import torch
import onnx
from pathlib import Path
from common import *
from eyelid_model import StableLids,NaturalWaveform,LIDS
from full_neutral_model import conditions
from fit import PAIRS

DEST=APP/'.local/mouth-tail-stable-v1'
PARENT=APP/'.local/eyelid-stable-v1'
FACE_PARENT=APP/'.local/expressive-full-neutral-v3'
DATA=APP/'.local/expressive-full-neutral-v1'
STEPS=1200


def load_net():
    with np.load(PARENT/'statistics.npz') as z:net=StableLids(FACE_PARENT,z['means'],z['scales'])
    net.load_state_dict(torch.load(PARENT/'face.pt',map_location='cpu',weights_only=True));return net


def main():
    torch.set_num_threads(4);torch.manual_seed(67);rng=np.random.default_rng(2067)
    DEST.mkdir(exist_ok=True)
    if (DEST/'selection.json').exists():raise ValueError('Completed mouth refinement exists')
    ds={}
    for split in ['train','dev','replay-train','replay-dev','common-dev']:
        with np.load(DATA/(split+'.npz')) as z:ds[split]={k:z[k].copy() for k in z.files}
    for split in ['train','dev']:
        d=ds[split];n=len(d['hidden']);following=np.arange(n);tail=np.zeros_like(d['valid'])
        totals={case:int(d['valid'][d['cases']==case].sum()) for case in set(d['cases'])};cursor={case:0 for case in totals}
        for i,case in enumerate(d['cases']):
            v=d['valid'][i];count=int(v.sum());tail[i,v]=np.arange(cursor[case],cursor[case]+count)>=totals[case]-48;cursor[case]+=count
            if i+11<n and case==d['cases'][i+11]:following[i]=i+11
        overlap=np.concatenate([d['valid'][:,25:30],d['valid'][following,:5]],axis=1);overlap[following==np.arange(n)]=False
        d['following']=following;d['overlap']=overlap;d['tail']=tail
    net=load_net();net.requires_grad_(False);net.parent.fast.requires_grad_(True)
    active=[c for c in range(52) if c not in [*LIDS,1,2,3,4,8,9,10,11,41,42,43,44,45]]
    with np.load(OUT/'calibration.npz') as z:
        gap=torch.from_numpy(z['gap_basis']);rest=torch.from_numpy(z['gap_neutral'])
        corner=torch.from_numpy(z['shape'][:,PAIRS[[0,-1]]].mean(2).reshape(52,6))
    train=ds['train'];scales=[]
    for e in range(11):
        ix=np.arange(e,len(train['target']),11);y=train['target'][ix][train['valid'][ix]];scales.append(np.maximum(y.std(0),.03))
    scale_table=torch.from_numpy(np.stack(scales))
    voices=[np.flatnonzero(train['voices']==v) for v in sorted(set(train['voices']))]
    def tensors(data,ix):
        h=torch.from_numpy(data['hidden'][ix]);b=torch.from_numpy(data['baseline'][ix])
        e=torch.from_numpy(data['emotion'][ix,None,:]).expand(-1,60,-1) if 'emotion' in data else torch.zeros(len(h),60,10)
        s=torch.zeros(len(h),60,3);s[:,:,1]=1
        return h,b,e,s
    def masked(x,v):return (x.square()*v[:,:,None]).sum()/(v.sum().clamp_min(1)*x.shape[-1])
    def gap_loss(p,g,v):
        predicted=p@gap+rest
        contacts=(g.median(-1).values<=.1)&v
        closure=(torch.relu(predicted.median(-1).values-.1).square()*contacts).sum()/(contacts.sum().clamp_min(1)*.05**2)
        return masked((predicted-g)/.1,v)+.1*closure
    def fit_loss(data,ix,p,e):
        y=torch.from_numpy(data['target'][ix]);v=torch.from_numpy(data['valid'][ix]);g=torch.from_numpy(data['gaps'][ix]);t=torch.from_numpy(data['tail'][ix])
        sigma=conditions(e[:,15:45])@scale_table
        coefficients=masked((p[:,:,active]-y[:,:,active])/sigma[:,:,active],v)
        delta=(p-y)@corner
        corners=masked(delta/.1,v)+2*masked(delta/.1,v&t)
        vel=(delta[:,1:]-delta[:,:-1])/.02
        velocity=masked(vel,v[:,1:]&v[:,:-1])
        return coefficients+32*gap_loss(p,g,v)+8*corners+2*velocity
    @torch.inference_mode()
    def cohort(data,ix,base=False):
        stats={}
        for start in range(0,len(ix),128):
            indices=ix[start:start+128];h,b,e,s=tensors(data,indices);p=b if base else net(h,b,e,s)
            predicted=(p[:,15:45]@gap+rest).numpy()
            for j,index in enumerate(indices):
                error=predicted[j,data['valid'][index]]-data['gaps'][index,data['valid'][index]]
                state=stats.setdefault(str(data['voices'][index]),[0.,0]);state[0]+=float(np.square(error).sum());state[1]+=error.size
        return float(np.mean([np.sqrt(a/max(1,b)) for a,b in stats.values()]))
    cohorts=[('neutral_dev',ds['dev'],np.arange(0,len(ds['dev']['hidden']),11)),('replay_dev',ds['replay-dev'],np.arange(len(ds['replay-dev']['hidden']))),('common_dev',ds['common-dev'],np.arange(len(ds['common-dev']['hidden'])))]
    limits={name:min(cohort(d,ix),cohort(d,ix,True))*1.05 for name,d,ix in cohorts}
    @torch.inference_mode()
    def evaluate():
        data=ds['dev'];score=0.;n=0
        for start in range(0,len(data['hidden']),128):
            ix=np.arange(start,min(start+128,len(data['hidden'])));h,b,e,s=tensors(data,ix);p=net(h,b,e,s)
            nh,nb,ne,ns=tensors(data,data['following'][ix]);npred=net(nh,nb,ne,ns)
            consistency=masked(((p[:,40:50]-npred[:,10:20])@corner)/.02,torch.from_numpy(data['overlap'][ix]))
            score+=float(fit_loss(data,ix,p[:,15:45],e)+8*consistency)*len(ix);n+=len(ix)
        gates={name:cohort(d,ix) for name,d,ix in cohorts}
        return score/n,gates
    protocol=dict(steps=STEPS,seed=67,parent_sha256=sha(PARENT/'model.onnx'),
        source_sha256=sha(__file__),data_sha256={n:sha(DATA/(n+'.npz')) for n in ds},
        target='Absolute teacher coefficients + fitted-teacher corner geometry; raw teacher lip-gap supervision',
        loss='coefficients +32 raw gap/closure +8 corner positions (3x tail weight) +2 corner velocity match +8 overlapping-window corner continuity +8 mouth replay',
        tail='Last .8 seconds of each training clip weighted; no runtime silence gate or forced-neutral pose',
        trained='Fast face branch only; brows, lids, gaze and compact speech core frozen',
        selection='Held-out voice dev loss with neutral cohort lip RMS <=1.05*min(parent,v4); at least 300 steps',cohort_limits_cm=limits)
    freeze(DEST/'protocol.json',protocol)
    params=list(net.parent.fast.parameters());optimizer=torch.optim.AdamW(params,lr=.0001,weight_decay=.001)
    history=[];best=float('inf');best_state=None;begun=time.monotonic()
    for step in range(1,STEPS+1):
        ix=np.concatenate([rng.choice(g,8,replace=True) for g in voices]);h,b,e,s=tensors(train,ix);p=net(h,b,e,s)
        nh,nb,ne,ns=tensors(train,train['following'][ix]);following=net(nh,nb,ne,ns)
        continuity=masked(((p[:,40:50]-following[:,10:20])@corner)/.02,torch.from_numpy(train['overlap'][ix]))
        replay=ds['replay-train'];ri=rng.integers(len(replay['hidden']),size=16);rh,rb,re,rs=tensors(replay,ri)
        rp=net(rh,rb,re,rs)[:,15:45];rloss=gap_loss(rp,torch.from_numpy(replay['gaps'][ri]),torch.from_numpy(replay['valid'][ri]))
        loss=fit_loss(train,ix,p[:,15:45],e)+8*continuity+8*rloss
        if not torch.isfinite(loss):raise FloatingPointError('Invalid mouth loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
        optimizer.param_groups[0]['lr']=float(.0001*(.1+.45*(1+np.cos(np.pi*step/STEPS))))
        if step%150==0:
            value,gates=evaluate();eligible=step>=300 and all(gates[n]<=limits[n] for n in gates)
            if eligible and value<best:best=value;best_step=step;best_state=copy.deepcopy(net.state_dict());torch.save(best_state,DEST/'best.pt')
            history.append(dict(step=step,dev_score=value,cohort_lip_rms_cm=gates,eligible=eligible,seconds=time.monotonic()-begun))
            (DEST/'progress.json').write_text(json.dumps(history,indent=2))
            print('mouth tails',step,round(value,3),'eligible',eligible,'seconds',round(time.monotonic()-begun),flush=True)
    if best_state is None:raise ValueError('No checkpoint passed lip-gap cohort gates')
    net.load_state_dict(best_state);net.eval();torch.save(best_state,DEST/'face.pt')
    model=NaturalWaveform(net).eval();style=torch.zeros(1,60,3);style[:,:,1]=1
    torch.onnx.export(model,(torch.zeros(1,16000),torch.zeros(1,60,10),style),str(DEST/'model.onnx'),input_names=['pcm','emotion','gaze_style'],output_names=['blendshapes','articulation'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(DEST/'model.onnx'))
    freeze(DEST/'selection.json',dict(best_step=best_step,dev_score=best,history=history,model_sha256=sha(DEST/'model.onnx')))
    print('EXPORTED',best_step,sha(DEST/'model.onnx'),flush=True)

if __name__=='__main__':main()

"""Train lid-only decoders; checkpoint selection uses held-out voice validation."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import onnx
from common import *
from eyelid_model import StableLids,NaturalWaveform,LIDS
from full_neutral_model import conditions

DEST=APP/'.local/eyelid-stable-v1'
PARENT=APP/'.local/expressive-full-neutral-v3'
DATA=APP/'.local/expressive-full-neutral-v1'
STEPS=2400

def main():
    torch.set_num_threads(4);torch.manual_seed(53);rng=np.random.default_rng(2053)
    DEST.mkdir(exist_ok=True)
    if (DEST/'selection.json').exists():raise ValueError('Completed lid run exists')
    ds={}
    for name in ['train','dev','test']:
        with np.load(DATA/(name+'.npz')) as z:ds[name]={k:z[k].copy() for k in ['hidden','target','valid','emotion','voices','cases']}
    train=ds['train'];means=[];scales=[]
    for e in range(11):
        ix=np.arange(e,len(train['target']),11);y=train['target'][ix][:,:,LIDS][train['valid'][ix]]
        means.append(y.mean(0));scales.append(np.maximum(y.std(0),.03))
    means=np.stack(means);scales=np.stack(scales);net=StableLids(PARENT,means,scales)
    parameters=list(net.slow.parameters())+list(net.blink.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=.001)
    table=torch.from_numpy(scales)
    groups=[np.flatnonzero(train['voices']==v) for v in sorted(set(train['voices']))]
    next_index=np.arange(len(train['hidden']))
    for i in range(len(next_index)-11):
        if train['cases'][i]==train['cases'][i+11]:next_index[i]=i+11
    overlap=np.concatenate([train['valid'][:,25:30],train['valid'][next_index,:5]],axis=1)
    overlap[next_index==np.arange(len(next_index))]=False
    def tensors(data,ix):
        h=torch.from_numpy(data['hidden'][ix]);e=torch.from_numpy(data['emotion'][ix,None,:]).expand(-1,60,-1)
        return h,e,torch.from_numpy(data['target'][ix][:,:,LIDS]),torch.from_numpy(data['valid'][ix])
    def masked(x,v):return (x.square()*v[:,:,None]).sum()/(v.sum().clamp_min(1)*x.shape[-1])
    def fit_loss(p,y,v,e):
        scale=conditions(e[:,15:45])@table;error=(p-y)/scale
        velocity=masked(error[:,1:]-error[:,:-1],v[:,1:]&v[:,:-1])
        slow=p[:,:,[1,2,4,5]];ss=scale[:,:,[1,2,4,5]]
        acceleration=masked(23.04*(slow[:,2:]-2*slow[:,1:-1]+slow[:,:-2])/ss[:,1:-1],v[:,2:]&v[:,1:-1]&v[:,:-2])
        return masked(error,v)+.2*velocity+.02*acceleration
    @torch.inference_mode()
    def score(data):
        loss=0.;n=0
        for start in range(0,len(data['hidden']),128):
            ix=np.arange(start,min(start+128,len(data['hidden'])));h,e,y,v=tensors(data,ix)
            p=net.lids(h,e)[:,15:45,LIDS];loss+=float(fit_loss(p,y,v,e))*len(ix);n+=len(ix)
        return loss/n
    protocol=dict(steps=STEPS,seed=53,parent_sha256=sha(PARENT/'model.onnx'),parent_face_sha256=sha(PARENT/'face.pt'),
        target='Absolute teacher eyelids in all eleven conditions; no neutral gate',
        trained_channels=LIDS,slow_anchors_per_second=6,blink_anchors_per_second=12,
        losses='within-condition standardized coefficient error + .2 velocity match + .02 slow-lid acceleration + overlap continuity',
        selection='Lowest held-out voice dev objective, at least 300 steps; test and James phrases excluded from selection',
        data_sha256={n:sha(DATA/(n+'.npz')) for n in ds},
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['eyelid_model.py','train_eyelids.py']})
    freeze(DEST/'protocol.json',protocol);best=float('inf');history=[];begun=time.monotonic()
    for step in range(1,STEPS+1):
        ix=np.concatenate([rng.choice(g,8,replace=True) for g in groups]);h,e,y,v=tensors(train,ix)
        all_lids=net.lids(h,e)[:,:,LIDS];p=all_lids[:,15:45]
        nh,ne,_,_=tensors(train,next_index[ix]);following=net.lids(nh,ne)[:,:,LIDS]
        scale=conditions(e[:,15:45])@table
        continuity=masked((all_lids[:,40:50]-following[:,10:20])/scale[:,:1],torch.from_numpy(overlap[ix]))
        loss=fit_loss(p,y,v,e)+continuity
        if not torch.isfinite(loss):raise FloatingPointError('Invalid eyelid loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        optimizer.param_groups[0]['lr']=float(.001*(.1+.45*(1+np.cos(np.pi*step/STEPS))))
        if step%150==0:
            value=score(ds['dev'])
            if step>=300 and value<best:best=value;best_step=step;best_state=copy.deepcopy(net.state_dict())
            history.append(dict(step=step,dev_score=value,seconds=time.monotonic()-begun))
            print('lids',step,'dev',round(value,4),'seconds',round(time.monotonic()-begun),flush=True)
    net.load_state_dict(best_state);net.eval();torch.save(net.state_dict(),DEST/'face.pt');np.savez(DEST/'statistics.npz',means=means,scales=scales)
    model=NaturalWaveform(net).eval();style=torch.zeros(1,60,3);style[:,:,1]=1
    torch.onnx.export(model,(torch.zeros(1,16000),torch.zeros(1,60,10),style),str(DEST/'model.onnx'),
        input_names=['pcm','emotion','gaze_style'],output_names=['blendshapes','articulation'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(DEST/'model.onnx'))
    freeze(DEST/'selection.json',dict(best_step=best_step,dev_score=best,test_score=score(ds['test']),history=history,model_sha256=sha(DEST/'model.onnx')))
    print('EXPORTED',best_step,sha(DEST/'model.onnx'),flush=True)

if __name__=='__main__':main()

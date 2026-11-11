"""Distill full neutral performance with a lip-gap non-regression gate."""
import copy
import json
import math
import time
import numpy as np
import torch
import onnx
from common import *
from gaze import NEW
from full_neutral_data import FULL
from full_neutral_model import NaturalFace,NaturalWaveform,conditions


def dataset(name):
    with np.load(FULL/f'{name}.npz') as z:return {k:z[k].copy() for k in z.files}


def tensors(data,indices):
    h=torch.from_numpy(data['hidden'][indices]);b=torch.from_numpy(data['baseline'][indices])
    if 'emotion' in data:e=torch.from_numpy(data['emotion'][indices,None,:]).expand(-1,60,-1)
    else:e=torch.zeros(len(indices),60,10)
    style=torch.zeros(len(indices),60,3);style[:,:,1]=1
    return h,b,e,style


def masked(error,valid):
    return (error.square()*valid[:,:,None]).sum()/(valid.sum().clamp_min(1)*error.shape[-1])


def main():
    torch.set_num_threads(4);torch.manual_seed(37);rng=np.random.default_rng(1037)
    if (FULL/'selection.json').exists():raise ValueError('Completed full-neutral model exists')
    data={s:dataset(s) for s in ['train','dev','test','replay-train','replay-dev','common-dev']}
    training=data['train'];means=[];scales=[]
    for e in range(11):
        ids=np.arange(e,len(training['target']),11);values=training['target'][ids][training['valid'][ids]]
        means.append(values.mean(0));scales.append(np.maximum(values.std(0),.03))
    means=np.stack(means);scales=np.stack(scales)
    adapter=torch.load(OUT/'adapter.pt',map_location='cpu',weights_only=True)
    gaze=torch.load(NEW/'gaze.pt',map_location='cpu',weights_only=True)
    net=NaturalFace(adapter,gaze,means[:,41:46],scales[:,41:46]);net.train()
    with np.load(OUT/'calibration.npz') as z:basis=torch.from_numpy(z['gap_basis']);rest=torch.from_numpy(z['gap_neutral'])
    scale_table=torch.from_numpy(scales)
    learn_ids=[i for i in range(52) if i not in [1,2,3,4,8,9,10,11]]
    voices=sorted(set(training['voices']));groups=[np.flatnonzero(training['voices']==v) for v in voices]
    next_index=np.arange(len(training['hidden']))
    for i in range(len(next_index)-11):
        if training['cases'][i]==training['cases'][i+11]:next_index[i]=i+11
    next_valid=np.concatenate([training['valid'][:,25:30],training['valid'][next_index,:5]],axis=1)
    next_valid[next_index==np.arange(len(next_index))]=False
    def gap_loss(prediction,truth,valid):
        predicted=prediction@basis+rest
        fit=masked((predicted-truth)/.1,valid)
        median=predicted.median(-1).values;target=truth.median(-1).values
        contacts=(target<=.1)&valid
        closure=(torch.relu(median-.1).square()*contacts).sum()/(contacts.sum().clamp_min(1)*.05**2)
        return fit+.1*closure
    @torch.inference_mode()
    def gap_rms(ds,indices=None,prior=False):
        if indices is None:indices=np.arange(len(ds['hidden']))
        stats={}
        for start in range(0,len(indices),128):
            ix=indices[start:start+128];h,b,e,s=tensors(ds,ix)
            p=b if prior else net(h,b,e,s)
            predicted=(p[:,15:45]@basis+rest).numpy()
            for j,index in enumerate(ix):
                key=str(ds['voices'][index]);v=ds['valid'][index];err=predicted[j,v]-ds['gaps'][index,v]
                stat=stats.setdefault(key,[0.,0]);stat[0]+=float(np.square(err).sum());stat[1]+=err.size
        per_voice={k:math.sqrt(s/max(1,n)) for k,(s,n) in stats.items()}
        return float(np.mean(list(per_voice.values()))),per_voice
    neutral_dev=np.arange(0,len(data['dev']['hidden']),11)
    baselines={}
    for name,ds,ids in [('neutral_dev',data['dev'],neutral_dev),('replay_dev',data['replay-dev'],None),('common_dev',data['common-dev'],None)]:
        rms,per_voice=gap_rms(ds,ids,True);baselines[name]=dict(rms=rms,per_voice=per_voice)
    protocol=dict(seed=37,steps=3000,batch_size=64,replay_batch=16,
        fast_learning_rate=.0003,brow_learning_rate=.001,brow_anchors_per_second=6,
        lip_gate='Every neutral validation cohort macro lip-gap RMS <= 1.05 times frozen finetune-r4 baseline',
        minimum_selection_step=300,selection='full-face coefficient + raw lip-gap + temporal validation objective, subject to lip gates',
        target='Absolute teacher performance in neutral and all emotion conditions; regional brow refit; no v4 neutral gate',
        losses='coefficient z-MSE (within-condition scales), 8*raw lip gaps + closure, .1 velocity-match, .05 brow acceleration, .5 brow motion variance, 1 brow overlap, 2 mouth replay',
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['full_neutral_model.py','train_full_neutral.py','full_neutral_data.py']},
        data_sha256={n:sha(FULL/f'{n}.npz') for n in data},baseline_sha256=sha(BASELINE),
        parent_sha256=sha(NEW/'fullface.onnx'),label_recipe_sha256=sha(FULL/'label-recipe.json'),
        baselines=baselines,head_rotation='deferred; official NVIDIA teacher only')
    freeze(FULL/'training-protocol.json',protocol)
    @torch.inference_mode()
    def evaluate():
        net.eval();ds=data['dev'];parts={};losses=[]
        for start in range(0,len(ds['hidden']),128):
            ix=np.arange(start,min(start+128,len(ds['hidden'])));h,b,e,s=tensors(ds,ix);p=net(h,b,e,s)[:,15:45]
            y=torch.from_numpy(ds['target'][ix]);v=torch.from_numpy(ds['valid'][ix]);g=torch.from_numpy(ds['gaps'][ix])
            sigma=(conditions(e[:,15:45])@scale_table)
            value=masked((p[:,:,learn_ids]-y[:,:,learn_ids])/sigma[:,:,learn_ids],v)+8*gap_loss(p,g,v)
            losses.append((float(value),len(ix)))
        for name,cohort,ids in [('neutral_dev',data['dev'],neutral_dev),('replay_dev',data['replay-dev'],None),('common_dev',data['common-dev'],None)]:
            rms,per_voice=gap_rms(cohort,ids);parts[name]=dict(rms=rms,per_voice=per_voice,ratio=rms/baselines[name]['rms'])
        score=sum(v*n for v,n in losses)/sum(n for _,n in losses)
        net.train();return score,parts
    optimizer=torch.optim.AdamW([{'params':net.fast.parameters(),'lr':.0003},
        {'params':net.brows.parameters(),'lr':.001}],weight_decay=.001)
    best=float('inf');best_step=None;best_state=None;history=[];begun=time.monotonic()
    for step in range(1,3001):
        ix=np.concatenate([rng.choice(g,8,replace=True) for g in groups]);h,b,e,s=tensors(training,ix)
        p=net(h,b,e,s);y=torch.from_numpy(training['target'][ix]);v=torch.from_numpy(training['valid'][ix]);g=torch.from_numpy(training['gaps'][ix])
        sigma=conditions(e[:,15:45])@scale_table;error=(p[:,15:45]-y)/sigma
        coefficient=masked(error[:,:,learn_ids],v)
        vel_valid=v[:,1:]&v[:,:-1]
        velocity=masked(error[:,1:,learn_ids]-error[:,:-1,learn_ids],vel_valid)
        brow=p[:,15:45,41:46];scale=sigma[:,:,41:46]
        triples=v[:,2:]&v[:,1:-1]&v[:,:-2]
        acceleration=masked(23.04*(brow[:,2:]-2*brow[:,1:-1]+brow[:,:-2])/scale[:,1:-1],triples)
        def window_std(values):
            n=v.sum(1).clamp_min(1)[:,None];mean=(values*v[:,:,None]).sum(1)/n
            variance=((values-mean[:,None])**2*v[:,:,None]).sum(1)/n
            return (variance+1e-8).sqrt()
        variance=((window_std(brow)-window_std(y[:,:,41:46]))/scale.mean(1)).square().mean()
        next_ix=next_index[ix];nh,_,ne,_=tensors(training,next_ix)
        following=net.brows(nh,ne)
        continuity=masked((p[:,40:50,41:46]-following[:,10:20])/scale[:,0:1],torch.from_numpy(next_valid[ix]))
        replay=data['replay-train'];ri=rng.integers(len(replay['hidden']),size=16);rh,rb,re,rs=tensors(replay,ri)
        rp=net(rh,rb,re,rs)[:,15:45]
        replay_loss=gap_loss(rp,torch.from_numpy(replay['gaps'][ri]),torch.from_numpy(replay['valid'][ri]))
        loss=coefficient+8*gap_loss(p[:,15:45],g,v)+.1*velocity+.05*acceleration+.5*variance+1.*continuity+2*replay_loss
        if not torch.isfinite(loss):raise FloatingPointError('Invalid full-neutral loss')
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.fast.parameters(),1.,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(net.brows.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        factor=.1+.45*(1+np.cos(np.pi*step/3000))
        optimizer.param_groups[0]['lr']=float(.0003*factor);optimizer.param_groups[1]['lr']=float(.001*factor)
        if step%150==0:
            score,parts=evaluate();eligible=step>=300 and all(r['ratio']<=1.05 for r in parts.values())
            if eligible and score<best:best,best_step,best_state=score,step,copy.deepcopy(net.state_dict())
            history.append(dict(step=step,score=score,lip_gates=parts,eligible=eligible,seconds=time.monotonic()-begun))
            (FULL/'progress.json').write_text(json.dumps(dict(best_step=best_step,history=history),indent=2))
            print('full neutral',step,'score',round(score,4),'lip ratios',[round(r['ratio'],3) for r in parts.values()],'best',best_step,'seconds',round(time.monotonic()-begun),flush=True)
    if best_state is None:raise ValueError('No trained checkpoint passed the predeclared lip-gap gates')
    net.load_state_dict(best_state);net.eval();torch.save(net.state_dict(),FULL/'face.pt')
    np.savez(FULL/'brow-statistics.npz',means=means[:,41:46],scales=scales[:,41:46])
    model=NaturalWaveform(net).eval();pcm=torch.zeros(1,16000);emotion=torch.zeros(1,60,10);style=torch.zeros(1,60,3);style[:,:,1]=1
    torch.onnx.export(model,(pcm,emotion,style),str(FULL/'model.onnx'),input_names=['pcm','emotion','gaze_style'],
        output_names=['blendshapes','articulation'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(FULL/'model.onnx'))
    freeze(FULL/'selection.json',dict(best_step=best_step,dev_score=best,history=history,bytes=(FULL/'model.onnx').stat().st_size,
        model_sha256=sha(FULL/'model.onnx'),protocol_sha256=sha(FULL/'training-protocol.json')))
    print('EXPORTED',best_step,sha(FULL/'model.onnx'),flush=True)


if __name__=='__main__':main()

"""Train a pilot adapter; select on development speech and export separately."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
import onnx
import onnxruntime as ort
from common import *
from model import ExpressionAdapter, ExpressiveWaveform


def load(split):
    with np.load(OUT/'features'/f'{split}.npz') as z:
        data={k:z[k].copy() for k in z.files}
    # Each audio window has neutral followed by the ten emotion conditions.
    # Protect the newly enabled lateral mouth/jaw shapes at neutral as well.
    if len(data['hidden'])%11:raise ValueError('Expected paired 11-condition windows')
    lateral=[15,16,21,22]
    neutral=data['target'][np.arange(len(data['target']))//11*11]
    if np.any(data['emotion'][::11]):raise ValueError('Neutral pairing changed')
    data['target'][:,:,lateral]=np.clip(data['target'][:,:,lateral]-neutral[:,:,lateral],0,1)
    data['protected']=np.union1d(data['active'],lateral)
    return data


def tensors(data,indices):
    return [torch.from_numpy(data[k][indices]) for k in ['hidden','baseline','emotion','target','valid']]


def prediction(net,hidden,speech,emotion):
    e=emotion[:,None,:].expand(-1,60,-1)
    return net(hidden,speech,e)


@torch.inference_mode()
def evaluate(net,data):
    accum={}
    for start in range(0,len(data['hidden']),128):
        indices=np.arange(start,min(start+128,len(data['hidden'])))
        h,b,e,y,m=tensors(data,indices)
        p=prediction(net,h,b,e)[:,15:45].numpy()
        error=p-data['target'][indices]
        for local,index in enumerate(indices):
            key=str(data['voices'][index])
            stat=accum.setdefault(key,np.zeros(53,np.float64))
            valid=data['valid'][index]
            stat[:52]+=np.square(error[local,valid]).sum(0)
            stat[52]+=valid.sum()
    channels=np.mean([np.sqrt(s[:52]/s[52]) for s in accum.values()],axis=0)
    return float(channels.mean()),channels.tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seed',type=int,default=37)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed+1000)
    train,dev=load('train'),load('dev')
    if (OUT/'selection.json').exists():raise ValueError('Completed model exists; use a new output directory')
    mean=train['hidden'].mean(axis=(0,1))
    scale=np.maximum(train['hidden'].std(axis=(0,1)),.1)
    targets=train['target'][train['valid']]
    output_scale=torch.from_numpy(np.maximum(targets.std(0),.05))
    net=ExpressionAdapter(train['protected'],mean,scale)
    initial,initial_channels=evaluate(net,dev)
    protocol=dict(seed=args.seed,steps=args.steps,batch_size=64,learning_rate=.0008,
        weight_decay=.001,selection='mean channel coefficient RMS, averaged equally over dev voices',
        loss='channel-scaled MSE + 0.1 scaled adjacent-frame velocity MSE',
        baseline_sha256=sha(BASELINE),label_recipe_sha256=sha(OUT/'label-recipe.json'),
        feature_sha256={s:sha(OUT/'features'/f'{s}.npz') for s in ['train','dev','test']},
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        target='Paired expression differences on 43 protected face channels (old 39 + lateral jaw/mouth); absolute teacher-derived gaze/tongue on remaining 9.',
        runtime='Torch CPU for adapter training; frozen source embeddings cached; no Player changes',
        head_motion='deferred; NVIDIA only; no DAWN labels')
    freeze(OUT/'training-protocol.json',protocol)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.001)
    best,best_step,best_state=initial,0,copy.deepcopy(net.state_dict())
    history=[]
    voices=sorted(set(train['voices']))
    groups=[np.flatnonzero(train['voices']==v) for v in voices]
    begun=time.monotonic()
    for step in range(1,args.steps+1):
        indices=np.concatenate([rng.choice(g,8,replace=True) for g in groups])
        h,b,e,y,m=tensors(train,indices)
        p=prediction(net,h,b,e)[:,15:45]
        err=(p-y)/output_scale
        loss=(err.square()*m[:,:,None]).sum()/(m.sum()*52)
        velocity=((p[:,1:]-p[:,:-1])-(y[:,1:]-y[:,:-1]))/output_scale
        pairs=m[:,1:]&m[:,:-1]
        loss=loss+.1*(velocity.square()*pairs[:,:,None]).sum()/(pairs.sum()*52)
        if not torch.isfinite(loss):raise FloatingPointError('Invalid training loss')
        optimizer.zero_grad()
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        lr=.00008+.5*(.0008-.00008)*(1+np.cos(np.pi*step/args.steps))
        for group in optimizer.param_groups:group['lr']=float(lr)
        if step%150==0 or step==args.steps:
            score,channels=evaluate(net,dev)
            if score<best:
                best,best_step,best_state=score,step,copy.deepcopy(net.state_dict())
            history.append(dict(step=step,dev_rms=score,channels=channels,loss=float(loss.detach()),seconds=time.monotonic()-begun))
            print('step',step,'dev RMS',round(score,5),'best',best_step,'seconds',round(time.monotonic()-begun),flush=True)
            (OUT/'progress.json').write_text(json.dumps(dict(initial=initial,best=best,best_step=best_step,history=history),indent=2))
    net.load_state_dict(best_state)
    net.eval()
    torch.save(net.state_dict(),OUT/'adapter.pt')
    model=ExpressiveWaveform(net).eval()
    pcm=torch.zeros((1,16000))
    emotion=torch.zeros((1,60,10))
    torch.onnx.export(model,(pcm,emotion),str(OUT/'model.part.onnx'),input_names=['pcm','emotion'],
                      output_names=['blendshapes'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(OUT/'model.part.onnx'))
    (OUT/'model.part.onnx').replace(OUT/'model.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(OUT/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    reference=ort.InferenceSession(str(BASELINE),sess_options=options,providers=['CPUExecutionProvider'])
    cases=[]
    for kind in ['silence','noise','speech']:
        if kind=='silence':audio=np.zeros((1,16000),np.float32)
        elif kind=='noise':audio=rng.normal(0,.02,(1,16000)).astype(np.float32)
        else:
            row=json.loads((OUT/'manifest.json').read_text())[0]
            audio=np.fromfile(row['audio'],'<f4')[:16000][None,:].copy()
        for emotion_index in [-1,0,1,6]:
            em=np.zeros((1,60,10),np.float32)
            if emotion_index>=0:em[:,:,emotion_index]=1
            with torch.inference_mode():native=model(torch.from_numpy(audio),torch.from_numpy(em)).numpy()
            exported=session.run(None,{'pcm':audio,'emotion':em})[0]
            error=float(np.max(np.abs(native-exported)))
            if error>3e-5 or not np.isfinite(exported).all():raise ValueError('ONNX parity failed')
            record=dict(audio=kind,emotion_index=emotion_index,native_onnx_max_error=error)
            if emotion_index<0:
                old=reference.run(None,{'pcm':audio})[0]
                drift=float(np.max(np.abs(exported[:,:,train['active']]-old[:,:,train['active']])))
                if drift>3e-5:raise ValueError('Neutral old-coefficient protection failed')
                record['neutral_old_39_max_drift']=drift
            cases.append(record)
    test=load('test')
    test_initial_net=ExpressionAdapter(train['protected'],mean,scale).eval()
    test_initial,test_initial_channels=evaluate(test_initial_net,test)
    test_final,test_final_channels=evaluate(net,test)
    report=dict(best_step=best_step,dev_initial_rms=initial,dev_selected_rms=best,
        test_initial_rms=test_initial,test_selected_rms=test_final,
        test_initial_channels=test_initial_channels,test_selected_channels=test_final_channels,
        adapter_parameters=sum(v.numel() for v in net.parameters()),
        total_predictor_parameters=331047+sum(v.numel() for v in net.parameters()),
        bytes=(OUT/'model.onnx').stat().st_size,model_sha256=sha(OUT/'model.onnx'),
        baseline_sha256=sha(BASELINE),protocol_sha256=sha(OUT/'training-protocol.json'),
        parity=cases,seconds=time.monotonic()-begun,
        limitations=['Single-seed small corpus pilot; no fresh independent confirmation.',
          'Emotion controls supplied as model inputs; this is not a trained audio-emotion classifier.',
          '25-degree gaze calibration and tongueStretch protrusion proxy need visual validation.',
          'No head rotation, browser WebGPU verification or 2.5D Player integration.'])
    freeze(OUT/'selection.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

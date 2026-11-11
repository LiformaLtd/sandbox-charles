"""Official NVIDIA multi-identity gaze labels and separately conditioned student."""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
from common import *
from fit import gaze_coefficients

NEW=Path(os.environ.get('A2F_FULLFACE_OUTPUT',APP/'.local/expressive-arkit-v3')).resolve()
STYLES=['Claire','James','Mark']


def labels():
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import animate,load_backend
    rows=json.loads((OUT/'manifest.json').read_text())
    freeze(NEW/'recipe.json',dict(parent_label_recipe_sha256=sha(OUT/'label-recipe.json'),
        teacher_sha256=sha(TEACHER/'models_cache/audio2face-v3/network.onnx'),
        source_sha256=sha(__file__),styles=STYLES,emotions=EMOTIONS,noise_seed=0,
        gaze_endpoint_degrees=25.,method='Direct 4D eye-rotation outputs; no skin fit or identity mixing.',
        head_motion='deferred; official NVIDIA teacher only; no DAWN'))
    recipe_sha=sha(NEW/'recipe.json')
    (NEW/'labels').mkdir(exist_ok=True)
    backend=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    started=time.monotonic()
    for index,row in enumerate(rows):
        if sha(row['audio'])!=row['sha256_audio']:raise ValueError('Changed source audio')
        pcm=np.fromfile(row['audio'],'<f4')
        for identity in STYLES:
            for ei,name in enumerate(['neutral',*EMOTIONS]):
                path=NEW/'labels'/f'{row["stem"]}--{name}--{identity}.npz'
                if path.exists():
                    with np.load(path) as z:
                        if str(z['recipe_sha256'])!=recipe_sha:raise ValueError('Changed gaze recipe')
                    continue
                if identity=='James':
                    with np.load(OUT/'labels'/f'{row["stem"]}--{name}.npz') as z:eyes=z['eyes'].copy()
                else:
                    emotion=np.zeros((1,30,10),np.float32)
                    if ei:emotion[:,:,ei-1]=1
                    def infer(inputs):return backend({**inputs,'emotion':emotion})
                    eyes=np.concatenate([g[:,-4:] for _,g in animate(pcm,infer,identity=identity,seed=0)])
                if eyes.shape!=((len(pcm)*60+15999)//16000,4) or not np.isfinite(eyes).all():
                    raise ValueError('Incomplete teacher gaze labels')
                np.savez_compressed(path,eyes=eyes,recipe_sha256=recipe_sha,audio_sha256=row['sha256_audio'])
        print('Gaze',index+1,'/',len(rows),'seconds',round(time.monotonic()-started,1),flush=True)


def features():
    for split in ['train','dev','test']:
        with np.load(OUT/'features'/f'{split}.npz') as z:cases=z['cases'];valid=z['valid']
        targets=np.zeros((len(cases),3,30,4),np.float32)
        cache={};cursors={}
        for i,case in enumerate(cases):
            key=str(case)
            if key not in cache:
                cache[key]=[np.load(NEW/'labels'/f'{key}--{identity}.npz')['eyes'] for identity in STYLES]
                cursors[key]=0
            start=cursors[key];count=int(valid[i].sum())
            for style,eyes in enumerate(cache[key]):targets[i,style,valid[i]]=eyes[start:start+count]/25
            cursors[key]+=count
        if any(cursors[k]!=len(cache[k][0]) for k in cache):raise ValueError('Incomplete gaze feature alignment')
        np.savez_compressed(NEW/f'{split}.npz',eyes=targets)
        print(split,'gaze windows',len(cases)*3,flush=True)
    # Count actual directional coverage across the official identity styles.
    report={}
    rows=json.loads((OUT/'manifest.json').read_text())
    names=json.loads((OUT/'channels.json').read_text())
    for split in ['train','dev','test']:
        eyes=np.concatenate([np.load(NEW/'labels'/f'{r["stem"]}--{name}--{identity}.npz')['eyes']
            for r in rows if r['split']==split for name in ['neutral',*EMOTIONS] for identity in STYLES])
        weights=gaze_coefficients(eyes);ids=[1,2,3,4,8,9,10,11]
        report[split]=dict(max={names[i]:float(weights[:,i].max()) for i in ids},
            std={names[i]:float(weights[:,i].std()) for i in ids},
            limited_range=[names[i] for i in ids if np.ptp(weights[:,i])<.05])
    freeze(NEW/'gaze-coverage.json',report)


def train():
    import torch
    import onnx
    from model import ExpressionAdapter
    from gaze_model import GazeHead,FullFaceWaveform
    from train import load
    torch.set_num_threads(4);torch.manual_seed(37)
    rng=np.random.default_rng(1037)
    data={s:load(s) for s in ['train','dev','test']}
    for split in data:
        with np.load(NEW/f'{split}.npz') as z:data[split]['eyes']=z['eyes'].copy()
    face_state=torch.load(OUT/'adapter.pt',map_location='cpu',weights_only=True)
    head=GazeHead(face_state['feature_mean'],face_state['feature_scale'])
    protocol=dict(steps=2400,seed=37,batch_size=64,styles=STYLES,loss='scaled signed-angle MSE + 0.1 adjacent velocity',
        lr=.001,face_checkpoint_sha256=sha(OUT/'adapter.pt'),parent_model_sha256=sha(OUT/'model.onnx'),
        recipe_sha256=sha(NEW/'recipe.json'),source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['gaze.py','gaze_model.py']},
        data_sha256={s:sha(NEW/f'{s}.npz') for s in data},
        note='Identity style controls gaze only. The James face adapter and speech predictor are frozen.')
    freeze(NEW/'training-protocol.json',protocol)
    if (NEW/'selection.json').exists():raise ValueError('Completed full-face trajectory exists')
    train=data['train'];target_scale=torch.from_numpy(np.maximum(train['eyes'].reshape(-1,4).std(0),.03))
    voices=sorted(set(train['voices']));groups=[np.flatnonzero(train['voices']==v) for v in voices]
    def batch(ds,indices,styles):
        hidden=torch.from_numpy(ds['hidden'][indices]);emotion=torch.from_numpy(ds['emotion'][indices,None,:]).expand(-1,60,-1)
        style=torch.nn.functional.one_hot(torch.as_tensor(styles),3).float()[:,None,:].expand(-1,60,-1)
        target=torch.from_numpy(ds['eyes'][indices,styles]);valid=torch.from_numpy(ds['valid'][indices])
        return hidden,emotion,style,target,valid
    @torch.inference_mode()
    def evaluate(ds):
        scores=[]
        for voice in sorted(set(ds['voices'])):
            selected=np.flatnonzero(ds['voices']==voice)
            for style in range(3):
                square,count=0.,0
                for start in range(0,len(selected),128):
                    ix=selected[start:start+128];h,e,s,y,m=batch(ds,ix,np.full(len(ix),style))
                    pred=head(h,e,s)[:,15:45];error=(pred-y)[m]
                    square+=float(error.square().sum());count+=error.numel()
                scores.append(dict(voice=str(voice),style=STYLES[style],rms_degrees=float(np.sqrt(square/count)*25)))
        return float(np.mean([r['rms_degrees'] for r in scores])),scores
    initial,_=evaluate(data['dev']);best,best_step,best_state=initial,0,copy.deepcopy(head.state_dict())
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    begun=time.monotonic();history=[]
    for step in range(1,2401):
        ix=np.concatenate([rng.choice(g,8,replace=True) for g in groups]);styles=rng.integers(3,size=len(ix))
        h,e,s,y,m=batch(train,ix,styles);p=head(h,e,s)[:,15:45]
        error=(p-y)/target_scale;loss=(error.square()*m[:,:,None]).sum()/(m.sum()*4)
        pairs=m[:,1:]&m[:,:-1]
        velocity=((p[:,1:]-p[:,:-1])-(y[:,1:]-y[:,:-1]))/target_scale
        loss=loss+.1*(velocity.square()*pairs[:,:,None]).sum()/(pairs.sum()*4)
        if not torch.isfinite(loss):raise FloatingPointError('Invalid gaze training loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        for group in optimizer.param_groups:group['lr']=float(.0001+.00045*(1+np.cos(np.pi*step/2400)))
        if step%150==0:
            score,_=evaluate(data['dev'])
            if score<best:best,best_step,best_state=score,step,copy.deepcopy(head.state_dict())
            history.append(dict(step=step,dev_rms_degrees=score,seconds=time.monotonic()-begun))
            print('Gaze step',step,'dev degrees',round(score,3),'best',best_step,'seconds',round(time.monotonic()-begun),flush=True)
    head.load_state_dict(best_state);head.eval();torch.save(head.state_dict(),NEW/'gaze.pt')
    test_score,test_styles=evaluate(data['test'])
    adapter=ExpressionAdapter(data['train']['protected'],np.zeros(96,np.float32),np.ones(96,np.float32))
    adapter.load_state_dict(face_state)
    model=FullFaceWaveform(adapter,head).eval()
    pcm=torch.zeros((1,16000));emotion=torch.zeros((1,60,10));style=torch.zeros((1,60,3));style[:,:,1]=1
    torch.onnx.export(model,(pcm,emotion,style),str(NEW/'model.part.onnx'),input_names=['pcm','emotion','gaze_style'],output_names=['blendshapes'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(NEW/'model.part.onnx'));(NEW/'model.part.onnx').replace(NEW/'model.onnx')
    report=dict(best_step=best_step,dev_initial_rms_degrees=initial,dev_selected_rms_degrees=best,
        test_rms_degrees=test_score,test_styles=test_styles,head_parameters=sum(v.numel() for v in head.parameters()),
        bytes=(NEW/'model.onnx').stat().st_size,model_sha256=sha(NEW/'model.onnx'),
        protocol_sha256=sha(NEW/'training-protocol.json'),face_checkpoint_sha256=sha(OUT/'adapter.pt'),
        limitation='Explicit emotion and gaze-style controls. No automatic emotion classifier or head-rotation outputs. Tongue remains a protrusion proxy.')
    freeze(NEW/'selection.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['labels','features','train']);args=parser.parse_args()
    NEW.mkdir(parents=True,exist_ok=True)
    {'labels':labels,'features':features,'train':train}[args.stage]()

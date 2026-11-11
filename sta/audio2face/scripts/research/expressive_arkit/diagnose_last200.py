"""Measure late mouth/chin movement and the model's zero-input response."""
import json
import os
import sys
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from infer_full import sequence
from infer import sequence as compact_sequence


def metrics(points):
    tail=points[-13:];steps=np.diff(tail,axis=0)
    return dict(last_200ms_step_rms_mm=float(np.sqrt(np.mean(steps**2))),
        last_200ms_path_mm=float(np.linalg.norm(steps,axis=-1).sum(0).mean()),
        last_200ms_net_rms_mm=float(np.sqrt(np.mean((tail[-1]-tail[0])**2))))


def main():
    public=APP/'public/assets/expressive';manifest=json.loads((public/'review.json').read_text());model=APP/'public'/manifest['candidate']['url'].lstrip('/')
    if sha(model)!=manifest['candidate']['sha256']:raise ValueError('Changed model')
    with np.load(OUT/'calibration.npz') as z:shape=z['shape'];neutral=z['neutral']
    with np.load(ASSETS/'model_data_James.npz') as z:teacher_neutral=z['neutral_skin']
    x,y,z=neutral.T
    regions=dict(mouth=np.flatnonzero((np.abs(x)<3)&(y>165.8)&(y<170.2)&(z>4.5)),chin=np.flatnonzero((np.abs(x)<2)&(y>162)&(y<166)&(z>3)))
    if min(len(r) for r in regions.values())<20:raise ValueError('Empty anatomical region')
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import animate,load_backend,MODEL_SHA256
    backend=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    out=APP/'.local/last200-diagnostic';out.mkdir(exist_ok=True)
    report=dict(model_sha256=sha(model),teacher_sha256=MODEL_SHA256,regions={k:r.tolist() for k,r in regions.items()},cases=[],silence=[])
    traces={};fig,axes=plt.subplots(5,2,figsize=(12,13),sharex=False)
    for row,sample in enumerate(manifest['samples']):
        audio_path=APP/'public'/sample['pcm'].lstrip('/');pcm=np.fromfile(audio_path,'<f4')
        if sha(audio_path)!=sample['sha256']:raise ValueError('Audio changed')
        v4=compact_sequence(BASELINE,pcm)
        for column,emotion in enumerate(['neutral','anger']):
            result=sequence(model,pcm,emotion);w=result['blendshapes'];count=len(w)
            control=np.zeros((1,30,10),np.float32)
            if emotion!='neutral':control[:,:,EMOTIONS.index(emotion)]=1
            def infer(inputs):return backend({**inputs,'emotion':control})
            raw={key:[] for key in regions}
            for _,geometry in animate(pcm,infer,identity='James',seed=0):
                skin=geometry[:,:72006].reshape(-1,24002,3)+teacher_neutral-neutral
                for key,region in regions.items():raw[key].append(skin[:,region]*10)
            case=dict(sample=sample['id'],emotion=emotion,frames=count,regions={},channels={})
            for key,region in regions.items():
                points=dict(candidate=np.einsum('fc,cvk->fvk',w,shape[:,region],optimize=False)*10,
                    v4=np.einsum('fc,cvk->fvk',v4,shape[:,region],optimize=False)*10,teacher=np.concatenate(raw[key]))
                case['regions'][key]={name:metrics(p) for name,p in points.items()}
                activity=np.sqrt(np.mean(np.diff(w[-13:],axis=0)**2,axis=0))*np.sqrt(np.mean(shape[:,region]**2,axis=(1,2)))*10
                case['channels'][key]=[{ 'name':manifest['channels'][int(i)],'motion_rms_mm':float(activity[i]) } for i in np.argsort(activity)[::-1][:8]]
                for name,p in points.items():traces[sample['id']+'--'+emotion+'--'+key+'--'+name]=p.mean(1)
            # Moving the window phase is diagnostic; it is not a runtime fix.
            shifted=sequence(model,np.pad(pcm,(4000,0)),emotion)['blendshapes'][15:15+count]
            case['quarter_second_prefix_chin_delta_mm']=float(np.sqrt(np.mean((np.einsum('fc,cvk->fvk',shifted[-13:]-w[-13:],shape[:,regions['chin']],optimize=False)*10)**2)))
            report['cases'].append(case)
            ax=axes[row,column]
            for name,label in [('teacher','Raw teacher'),('v4','Released v4'),('candidate','Current candidate')]:
                p=traces[sample['id']+'--'+emotion+'--chin--'+name]
                ax.plot(np.arange(len(p))/60,p[:,1],label=label)
            ax.axvspan(max(0,count/60-.2),count/60,alpha=.08,color='red');ax.set_xlim(max(0,count/60-.65),count/60)
            ax.set_title(sample['id']+' · '+emotion,fontsize=9);ax.set_ylabel('Mean chin Y (mm)');ax.grid(alpha=.2)
            print(sample['id'],emotion,'chin step raw/v4/new', [round(case['regions']['chin'][n]['last_200ms_step_rms_mm'],6) for n in ['teacher','v4','candidate']],flush=True)
    for emotion in ['neutral','anger']:
        pcm=np.zeros(48000,np.float32);w=sequence(model,pcm,emotion)['blendshapes'];v4=compact_sequence(BASELINE,pcm)
        item=dict(emotion=emotion,period_30_frame_coefficient_delta=float(np.abs(w[15:-30]-w[45:]).max()),regions={})
        for key,region in regions.items():
            p=np.einsum('fc,cvk->fvk',w,shape[:,region],optimize=False)*10;q=np.einsum('fc,cvk->fvk',v4,shape[:,region],optimize=False)*10
            item['regions'][key]=dict(candidate_steady_step_rms_mm=float(np.sqrt(np.mean(np.diff(p[30:],axis=0)**2))),v4_steady_step_rms_mm=float(np.sqrt(np.mean(np.diff(q[30:],axis=0)**2))))
            traces['silence--'+emotion+'--'+key+'--candidate']=p.mean(1)
        report['silence'].append(item);print('silence',item,flush=True)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'last200-chin.png',dpi=150);plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'traces.npz',**traces)

if __name__=='__main__':main()

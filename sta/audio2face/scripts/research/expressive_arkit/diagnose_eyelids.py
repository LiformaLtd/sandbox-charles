"""Compare eyelid vibration with raw teacher geometry on all James fixtures."""
import argparse
import json
import os
import sys
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
from common import *
from gaze import NEW
from fit import FullFaceFit
from infer_full import sequence
from infer import sequence as compact_sequence
from diagnose_brows import temporal_metrics


def main(candidate_path=None,output=None):
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
    previous=APP/'public'/manifest['candidate']['url'].lstrip('/')
    if sha(previous)!=manifest['candidate']['sha256']:raise ValueError('Staged candidate SHA mismatch')
    candidate=candidate_path or previous
    out=output or APP/'.local/eyelid-diagnostic';out.mkdir(exist_ok=True)
    fit=FullFaceFit(ASSETS,WORK/'arkit-student/james_arkit_v3.ma',APP)
    ids=np.array([0,5,6,7,12,13])
    region=np.unique(np.concatenate([np.argsort(np.linalg.norm(fit.shape[c],axis=1))[-256:] for c in ids]))
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import load_backend,animate,MODEL_SHA256
    backend=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    report=dict(candidate_sha256=sha(candidate),teacher_sha256=MODEL_SHA256,
        method='Union of 256 strongest vertices for each of six eyelid channels; XYZ RMS in mm. Hann periodogram, linear detrend, exclude first/last 250 ms. Eyelid-only reconstruction separates lid coefficients from other face motion. Fast motion is diagnostic, not automatically erroneous: true blinks are fast.',
        channels=[fit.names[i] for i in ids],region_vertices=len(region),cases=[])
    traces={}
    for sample in manifest['samples']:
        audio_path=APP/'public'/sample['pcm'].lstrip('/')
        if sha(audio_path)!=sample['sha256']:raise ValueError('Audio changed')
        audio=np.fromfile(audio_path,'<f4')
        compact=compact_sequence(BASELINE,audio)
        for emotion in ['neutral','anger']:
            control=np.zeros((1,30,10),np.float32)
            if emotion!='neutral':control[:,:,EMOTIONS.index(emotion)]=1
            def infer(inputs):return backend({**inputs,'emotion':control})
            weights=[];raw=[]
            for _,geometry in animate(audio,infer,identity='James',seed=0):
                w,_=fit(geometry);weights.append(w)
                skin=geometry[:,:72006].reshape(-1,24002,3)
                raw.append((skin[:,region]+fit.model_neutral[region]-fit.neutral[region])*10)
            coeff=dict(teacher_fit=np.concatenate(weights),current_v4=compact,
                original_candidate=sequence(NEW/'fullface.onnx',audio,emotion,1.,'James')['blendshapes'],
                previous_candidate=sequence(previous,audio,emotion,1.,'James')['blendshapes'],
                new_candidate=sequence(candidate,audio,emotion,1.,'James')['blendshapes'])
            full=dict(teacher_raw=temporal_metrics(np.concatenate(raw)));lid={};channels={}
            for name,w in coeff.items():
                full[name]=temporal_metrics(np.einsum('fc,cvk->fvk',w,fit.shape[:,region],optimize=False)*10)
                lid[name]=temporal_metrics(np.einsum('fc,cvk->fvk',w[:,ids],fit.shape[ids][:,region],optimize=False)*10)
                channels[name]={fit.names[c]:temporal_metrics(w[:,c:c+1]) for c in ids}
                traces[sample['id']+'--'+emotion+'--'+name]=w[:,ids]
            report['cases'].append(dict(sample=sample['id'],emotion=emotion,full_surface=full,lid_only_surface=lid,channels=channels))
            print(sample['id'],emotion,'raw/fit/v4/old/new fast mm',
                [round(full[n]['fast_4_15_hz_rms'],4) for n in ['teacher_raw','teacher_fit','current_v4','original_candidate','new_candidate']],flush=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'tracks.npz',**traces)
    print(out/'metrics.json',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();main(args.candidate,args.output)

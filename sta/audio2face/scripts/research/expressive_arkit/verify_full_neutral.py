"""Matched teacher geometry, eyebrow jitter and lip-gap RMS on the James clips."""
import json
import os
import sys
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
import torch
import onnx
import onnxruntime as ort
from common import *
from gaze import NEW
from full_neutral_data import FULL
from full_neutral_model import NaturalFace,NaturalWaveform
from infer_full import sequence
from diagnose_brows import temporal_metrics


def native_model():
    adapter=torch.load(OUT/'adapter.pt',map_location='cpu',weights_only=True)
    gaze=torch.load(NEW/'gaze.pt',map_location='cpu',weights_only=True)
    with np.load(FULL/'brow-statistics.npz') as z:means=z['means'];scales=z['scales']
    face=NaturalFace(adapter,gaze,means,scales)
    face.load_state_dict(torch.load(FULL/'face.pt',map_location='cpu',weights_only=True))
    return NaturalWaveform(face).eval()


def main():
    torch.set_num_threads(4)
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import load_backend,animate,MODEL_SHA256
    from fit import PAIRS
    onnx.checker.check_model(onnx.load(FULL/'model.onnx'))
    upper,lower=PAIRS.T
    with np.load(OUT/'calibration.npz') as z:shape=z['shape'];rest=z['gap_neutral'];basis=z['gap_basis'];neutral=z['neutral']
    with np.load(ASSETS/'model_data_James.npz') as z:raw_neutral=z['neutral_skin']
    region=np.argsort(np.max(np.linalg.norm(shape[41:46],axis=2),axis=0))[-512:]
    backend=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    native=native_model();options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(FULL/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    prior_session=ort.InferenceSession(str(NEW/'fullface.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
    phrases=[];parity=[];tracks={};references={};violations=[]
    for sample in manifest['samples']:
        path=APP/'public'/sample['pcm'].lstrip('/');pcm=np.fromfile(path,'<f4')
        if sha(path)!=sample['sha256']:raise ValueError('Changed demo audio')
        for emotion in ['neutral','anger']:
            control=np.zeros((1,30,10),np.float32)
            if emotion!='neutral':control[:,:,EMOTIONS.index(emotion)]=1
            def infer(inputs):return backend({**inputs,'emotion':control})
            gaps=[];brows=[]
            for indices,g in animate(pcm,infer,identity='James',seed=0):
                skin=g[:,:72006].reshape(-1,24002,3)+raw_neutral
                gaps.append(skin[:,upper,1]-skin[:,lower,1]);brows.append((skin[:,region]-neutral[region])*10)
            truth_gap=np.concatenate(gaps);truth_brow=np.concatenate(brows)
            before=sequence(NEW/'fullface.onnx',pcm,emotion,1.,'James')
            after=sequence(FULL/'model.onnx',pcm,emotion,1.,'James')
            a,b=before['blendshapes'],after['blendshapes']
            if a.shape!=b.shape or len(a)!=len(truth_gap):raise ValueError('Audio/animation timestamp mismatch')
            if not np.array_equal(before['articulation'],after['articulation']):
                # Export precision may differ while all values remain within the
                # established native/ONNX tolerance. It is a reference output,
                # not the active mouth constraint in this revised experiment.
                if np.abs(before['articulation']-after['articulation']).max()>3e-5:raise ValueError('Speech-core export drift')
            ga=np.einsum('fc,ck->fk',a,basis,optimize=False)+rest
            gb=np.einsum('fc,ck->fk',b,basis,optimize=False)+rest
            old_rms=float(np.sqrt(np.mean((ga-truth_gap)**2)))*10
            new_rms=float(np.sqrt(np.mean((gb-truth_gap)**2)))*10
            sa=np.einsum('fc,cvk->fvk',a,shape[:,region],optimize=False)*10
            sb=np.einsum('fc,cvk->fvk',b,shape[:,region],optimize=False)*10
            ma,mb,mt=temporal_metrics(sa),temporal_metrics(sb),temporal_metrics(truth_brow)
            actual=np.median(truth_gap,axis=1)<=.1
            def contact_f1(g):
                predicted=np.median(g,axis=1)<=.1
                return float(2*(actual&predicted).sum()/max(1,actual.sum()+predicted.sum()))
            ratio=new_rms/max(old_rms,1e-12)
            if ratio>1.05:violations.append(sample['id']+' '+emotion+' lip RMS >5%')
            if emotion=='anger' and mb['fast_4_15_hz_rms']>.35*ma['fast_4_15_hz_rms']:
                violations.append(sample['id']+' insufficient jitter reduction')
            row=dict(sample=sample['id'],emotion=emotion,frames=len(a),old_lip_rms_mm=old_rms,new_lip_rms_mm=new_rms,
                lip_ratio=ratio,old_contact_f1=contact_f1(ga),new_contact_f1=contact_f1(gb),
                before_brows=ma,after_brows=mb,teacher_brows=mt,
                old_teacher_brow_rms_mm=float(np.sqrt(np.mean((sa-truth_brow)**2))),
                new_teacher_brow_rms_mm=float(np.sqrt(np.mean((sb-truth_brow)**2))))
            phrases.append(row)
            key=sample['id']+'--'+emotion;tracks[key+'--before']=a;tracks[key+'--after']=b;references[key]=truth_gap
            # Browser-quality reference bank uses the exact same raw geometry.
            row['reference_key']=key
        for ei in [-1,1,6,0]:
            for style_index in range(3):
                e=np.zeros((1,60,10),np.float32)
                if ei>=0:e[:,:,ei]=1
                s=np.zeros((1,60,3),np.float32);s[:,:,style_index]=1
                inputs=dict(pcm=pcm[:16000][None,:].copy(),emotion=e,gaze_style=s)
                with torch.inference_mode():ref=native(*(torch.from_numpy(inputs[n]) for n in ['pcm','emotion','gaze_style']))
                result=session.run(None,inputs)
                error=max(float(np.abs(r-t.numpy()).max()) for r,t in zip(result,ref))
                if error>3e-5 or any(not np.isfinite(r).all() or r.min()<0 or r.max()>1 for r in result):raise ValueError('Native/ONNX validity check failed')
                parity.append(dict(sample=sample['id'],emotion_index=ei,gaze_style=style_index,max_error=error))
    protected=json.loads((OUT/'protected-before.json').read_text());changed=[p for p,h in protected.items() if sha(p)!=h]
    if changed:raise ValueError('Protected primary files changed')
    np.savez_compressed(FULL/'phrase-tracks.npz',**tracks)
    np.savez_compressed(FULL/'raw-gap-references.npz',**references)
    report=dict(model_sha256=sha(FULL/'model.onnx'),bytes=(FULL/'model.onnx').stat().st_size,
        teacher_sha256=MODEL_SHA256,phrases=phrases,parity=parity,violations=violations,
        passed=not violations,lip_tolerance=1.05,protected_primary_files_checked=len(protected),
        neutral='Absolute teacher targets for every learned facial channel; no v4 neutral gate',
        limitations=['Approximation in the James ARKit rig, not exact reproduction of all dense teacher geometry.',
          'Head rotation deferred; teacher-internal tongue deformations beyond ARKit tongueOut are not represented.',
          'No fresh independent speech confirmation or Player/2.5D rollout.'])
    (FULL/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if violations:raise ValueError('Regression gates failed: '+str(violations))


if __name__=='__main__':main()

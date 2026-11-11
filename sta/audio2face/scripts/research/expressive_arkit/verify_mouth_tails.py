"""Verify corner-tail fidelity/continuity and lip-gap regressions before review."""
import json
import sys
import os
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
import torch
import onnxruntime as ort
from common import *
from refine_mouth_tails import DEST,PARENT,load_net,NaturalWaveform
from eyelid_model import LIDS
from fit import PAIRS
from infer_full import sequence


def main():
    torch.set_num_threads(4)
    net=load_net();net.load_state_dict(torch.load(DEST/'face.pt',map_location='cpu',weights_only=True));net.eval();native=NaturalWaveform(net).eval()
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
    with np.load(OUT/'calibration.npz') as z:basis=z['gap_basis'];rest=z['gap_neutral'];corner=z['shape'][:,PAIRS[[0,-1]]].mean(2)
    cached=np.load(APP/'.local/mouth-tail-diagnostic/traces.npz')
    diagnostic=json.loads((APP/'.local/mouth-tail-diagnostic/metrics.json').read_text())
    if diagnostic['candidate_sha256']!=sha(PARENT/'model.onnx'):raise ValueError('Diagnostic parent changed')
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import load_backend,animate
    backend=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    with np.load(ASSETS/'model_data_James.npz') as z:teacher_neutral=z['neutral_skin']
    older=json.loads((PARENT/'verification.json').read_text())
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(DEST/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    frozen=[*LIDS,1,2,3,4,8,9,10,11,41,42,43,44,45];failures=[];known=[];phrases=[];references={};parity=[];traces={};joins_before=[];joins_after=[]
    for sample in manifest['samples']:
        pcm=np.fromfile(APP/'public'/sample['pcm'].lstrip('/'),'<f4')
        for emotion in ['neutral','anger']:
            before=sequence(PARENT/'model.onnx',pcm,emotion,1.,'James');after=sequence(DEST/'model.onnx',pcm,emotion,1.,'James')
            a=before['blendshapes'];b=after['blendshapes'];n=len(a)
            if a.shape!=b.shape:raise ValueError('Timeline changed')
            stable=float(np.abs(a[:,frozen]-b[:,frozen]).max())
            if stable>3e-5:failures.append('Brow/lid/gaze coefficients changed')
            if np.abs(before['articulation']-after['articulation']).max()>3e-5:failures.append('Frozen speech reference changed')
            key=sample['id']+'--'+emotion;truth_corner=cached[key+'--teacher_raw--points']
            ca=np.einsum('fc,cpk->fpk',a,corner,optimize=False)*10;cb=np.einsum('fc,cpk->fpk',b,corner,optimize=False)*10
            start=max(0,n-48)
            old_corner=float(np.sqrt(np.mean((ca[start:]-truth_corner[start:])**2)));new_corner=float(np.sqrt(np.mean((cb[start:]-truth_corner[start:])**2)))
            join=(np.arange(1,n)-15)%30==0;join&=np.arange(1,n)>=start
            da=np.diff(ca,axis=0)-np.diff(truth_corner,axis=0);db=np.diff(cb,axis=0)-np.diff(truth_corner,axis=0)
            joins_before.extend(np.square(da[join]).ravel());joins_after.extend(np.square(db[join]).ravel())
            control=np.zeros((1,30,10),np.float32)
            if emotion!='neutral':control[:,:,EMOTIONS.index(emotion)]=1
            def infer(inputs):return backend({**inputs,'emotion':control})
            gaps=[]
            for _,g in animate(pcm,infer,identity='James',seed=0):
                skin=g[:,:72006].reshape(-1,24002,3)+teacher_neutral
                gaps.append(skin[:,PAIRS[:,0],1]-skin[:,PAIRS[:,1],1])
            truth=np.concatenate(gaps);references[key]=truth
            ga=np.einsum('fc,ck->fk',a,basis,optimize=False)+rest;gb=np.einsum('fc,ck->fk',b,basis,optimize=False)+rest
            old_gap=float(np.sqrt(np.mean((ga-truth)**2)))*10;new_gap=float(np.sqrt(np.mean((gb-truth)**2)))*10
            prior=next((r for r in older['phrases'] if r['sample']==sample['id'] and r['emotion']==emotion),None)
            reference_rms=prior['old_lip_rms_mm'] if prior else old_gap
            limit=min(old_gap*1.05,max(reference_rms*1.05,old_gap))
            if new_gap>limit+1e-4:failures.append(key+' lip gap regression')
            if new_gap>reference_rms*1.05:known.append(key+' lip RMS >5%')
            if new_corner>old_corner*1.1:failures.append(key+' tail corner error >10% worse')
            if sample['id']=='test-supertonic-f5-000' and emotion=='neutral' and new_corner>old_corner*.95:failures.append('F5 neutral tail corner gain <5%')
            phrases.append(dict(sample=sample['id'],emotion=emotion,reference_key=key,frames=n,old_lip_rms_mm=reference_rms,new_lip_rms_mm=new_gap,lip_ratio=new_gap/reference_rms,
                parent_lip_rms_mm=old_gap,lip_regression_limit_mm=limit,tail_corner_error_before_mm=old_corner,tail_corner_error_after_mm=new_corner,
                frozen_brow_lid_gaze_max_drift=stable))
            traces[key+'--before']=ca;traces[key+'--after']=cb;traces[key+'--teacher']=truth_corner
        for ei in [-1,1,6,0]:
            for si in range(3):
                e=np.zeros((1,60,10),np.float32)
                if ei>=0:e[:,:,ei]=1
                style=np.zeros((1,60,3),np.float32);style[:,:,si]=1
                inputs=dict(pcm=pcm[:16000][None,:].copy(),emotion=e,gaze_style=style)
                with torch.inference_mode():expected=native(*(torch.from_numpy(inputs[name]) for name in ['pcm','emotion','gaze_style']))
                actual=session.run(None,inputs);error=max(float(np.abs(a-b.numpy()).max()) for a,b in zip(actual,expected))
                if error>3e-5 or any(not np.isfinite(a).all() or a.min()<0 or a.max()>1 for a in actual):failures.append('Native/ONNX parity or bounds')
                parity.append(dict(sample=sample['id'],emotion_index=ei,gaze_style=si,max_error=error))
    gain=np.mean([r['tail_corner_error_after_mm'] for r in phrases])/np.mean([r['tail_corner_error_before_mm'] for r in phrases])
    join_ratio=float(np.sqrt(np.mean(joins_after)/np.mean(joins_before)))
    if gain>.9:failures.append('Mean tail corner gain <10%')
    if join_ratio>.8:failures.append('Tail join error gain <20%')
    protected=json.loads((OUT/'protected-before.json').read_text())
    if any(sha(p)!=h for p,h in protected.items()):failures.append('Protected primary files changed')
    report=dict(model_sha256=sha(DEST/'model.onnx'),parent_sha256=sha(PARENT/'model.onnx'),bytes=(DEST/'model.onnx').stat().st_size,
        passed=not(failures or known),mouth_tail_checks_passed=not failures,violations=known+failures,mouth_tail_failures=failures,
        lip_tolerance=1.05,lip_numerical_tolerance_mm=1e-4,phrases=phrases,parity=parity,mean_tail_corner_error_ratio=float(gain),tail_join_error_ratio=join_ratio,
        protected_primary_files_checked=len(protected),limitations=['Reused known research recordings; no new independent release confirmation.','Raw teacher mouth-corner settling is retained; no runtime forced-neutral/silence gate.','Player and 2.5D remain unchanged.'])
    np.savez_compressed(DEST/'raw-gap-references.npz',**references);np.savez_compressed(DEST/'corner-traces.npz',**traces)
    (DEST/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if failures:raise ValueError('Mouth-tail checks failed: '+str(failures))

if __name__=='__main__':main()

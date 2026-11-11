"""Export/check learned quiet-tail correction against the previously reviewed graph.

Acceptance is fixed before running: every phrase/control lip and corner error
within 5%; every final-200ms chin step <=75%, pooled <=50%; silence <=50%;
25 other controls/articulation unchanged; upper-face fast motion <=110%.
These are reused development recordings, not an independent release benchmark.
"""
import json
import argparse
import shutil
import numpy as np
import torch
import onnx
import onnxruntime as ort
from common import *
from speech_rest_energy_model import CURRENT, DEST, EnergyRestFace, EnergyRestWaveform
from infer_full import sequence
from fit import PAIRS
from diagnose_brows import temporal_metrics


def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def project(w,b):return np.einsum('fc,cvk->fvk',w,b,optimize=False)*10

def main(directory=None,kind='learned'):
    global DEST
    if directory is not None:DEST=directory
    torch.set_num_threads(4)
    if kind=='weighted':
        from weighted_rest_model import WeightedRestWaveform
        native=WeightedRestWaveform().eval();DEST.mkdir(exist_ok=True)
    elif kind=='adaptive':
        from adaptive_rest_model import AdaptiveRestWaveform
        native=AdaptiveRestWaveform();native.load_rest(DEST/'wave.pt');native.eval()
    elif kind=='conditioned':
        from conditioned_rest_model import ConditionedRestWaveform
        native=ConditionedRestWaveform();native.load_state_dict(torch.load(DEST/'wave.pt',map_location='cpu',weights_only=True));native.eval()
    else:
        face=EnergyRestFace();face.load_state_dict(torch.load(DEST/'face.pt',map_location='cpu',weights_only=True));native=EnergyRestWaveform(face).eval()
    model=DEST/'model.onnx';style=torch.zeros(1,60,3);style[:,:,1]=1
    if not model.exists():
        torch.onnx.export(native,(torch.zeros(1,16000),torch.zeros(1,60,10),style),str(model),input_names=['pcm','emotion','gaze_style'],output_names=['blendshapes','articulation'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(model))
    print('EXPORTED',sha(model),model.stat().st_size,flush=True)
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
    with np.load(OUT/'calibration.npz') as z:shape=z['shape'];gap=z['gap_basis'];rest=z['gap_neutral'];neutral=z['neutral']
    x,y,z=neutral.T
    regions=dict(mouth=np.flatnonzero((np.abs(x)<3)&(y>165.8)&(y<170.2)&(z>4.5)),chin=np.flatnonzero((np.abs(x)<2)&(y>162)&(y<166)&(z>3)),
        brows=np.argsort(np.max(np.linalg.norm(shape[41:46],axis=2),axis=0))[-512:],
        lids=np.unique(np.concatenate([np.argsort(np.linalg.norm(shape[c],axis=1))[-256:] for c in [0,5,6,7,12,13]])))
    corner=shape[:,PAIRS[[0,-1]]].mean(2)
    refs=np.load(CURRENT/'raw-gap-references.npz');cornerrefs=np.load(APP/'.local/mouth-tail-diagnostic/traces.npz')
    previous=json.loads((CURRENT/'verification.json').read_text())
    fixed=list(range(14))+list(range(41,52));failures=[];known=[];rows=[];traces={};silence=[];parity=[]
    options=ort.SessionOptions();options.intra_op_num_threads=4;session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    for sample in manifest['samples']:
        pcm=np.fromfile(APP/'public'/sample['pcm'].lstrip('/'),'<f4')
        for emotion in ['neutral','anger']:
            a=sequence(CURRENT/'model.onnx',pcm,emotion);b=sequence(model,pcm,emotion);wa=a['blendshapes'];wb=b['blendshapes'];key=sample['id']+'--'+emotion
            ga=(np.einsum('fc,ck->fk',wa,gap,optimize=False)+rest-refs[key])*10;gb=(np.einsum('fc,ck->fk',wb,gap,optimize=False)+rest-refs[key])*10
            ca=project(wa,corner);cb=project(wb,corner);truth=cornerrefs[key+'--teacher_raw--points']
            oldcorner=rms(ca[-48:]-truth[-48:]);newcorner=rms(cb[-48:]-truth[-48:]);oldgap=rms(ga);newgap=rms(gb)
            prior=next(r for r in previous['phrases'] if r['reference_key']==key);legacy=prior['old_lip_rms_mm']
            row=dict(sample=sample['id'],emotion=emotion,reference_key=key,frames=len(wa),old_lip_rms_mm=legacy,new_lip_rms_mm=newgap,lip_ratio=newgap/legacy,
                previous_review_lip_rms_mm=oldgap,lip_ratio_vs_previous_review=newgap/oldgap,lip_regression_limit_mm=oldgap*1.05,
                tail_corner_error_before_mm=oldcorner,tail_corner_error_after_mm=newcorner,regions={},upper_face={})
            if newgap>oldgap*1.05+1e-4:failures.append(key+' lip RMS >5% versus current')
            if newcorner>oldcorner*1.05+1e-4:failures.append(key+' corner error >5% versus current')
            if newgap>legacy*1.05+1e-4:known.append(key+' lip RMS >5% versus original reference')
            row['frozen_other_25_max_drift']=float(np.abs(wa[:,fixed]-wb[:,fixed]).max())
            if row['frozen_other_25_max_drift']>3e-5 or np.abs(a['articulation']-b['articulation']).max()>3e-5:failures.append(key+' frozen output changed')
            for name,region in regions.items():
                pa=project(wa,shape[:,region]);pb=project(wb,shape[:,region])
                if name in ['chin','mouth']:
                    sa=rms(np.diff(pa[-13:],axis=0));sb=rms(np.diff(pb[-13:],axis=0))
                    row['regions'][name]=dict(before_step_mm=sa,after_step_mm=sb,ratio=sb/sa)
                    traces[key+'--'+name+'--before']=pa.mean(1);traces[key+'--'+name+'--after']=pb.mean(1)
                    if name=='chin' and sb>sa*.75:failures.append(key+' final200ms chin reduction <25%')
                    if name=='mouth' and sb>sa*1.05:failures.append(key+' final200ms mouth motion >5%')
                else:
                    fa=temporal_metrics(pa)['fast_4_15_hz_rms'];fb=temporal_metrics(pb)['fast_4_15_hz_rms'];row['upper_face'][name]=dict(before=fa,after=fb,ratio=fb/fa)
                    if fb>fa*1.1:failures.append(key+' '+name+' fast motion >10%')
            # Appending audio cannot be used as a runtime end-of-clip switch.
            extended=sequence(model,np.pad(pcm,(0,16000)),emotion)['blendshapes'][:len(wb)]
            # Ignore last 0.5s, whose context genuinely changes on padding.
            row['extended_audio_earlier_max_drift']=float(np.abs(extended[:-30]-wb[:-30]).max())
            if row['extended_audio_earlier_max_drift']>3e-5:failures.append(key+' earlier audio changed with extension')
            rows.append(row);print(key,'chin',round(row['regions']['chin']['ratio'],3),'mouth',round(row['regions']['mouth']['ratio'],3),'gap',round(newgap/oldgap,3),'corner',round(newcorner/oldcorner,3),flush=True)
        for ei in [-1,1,6,0]:
            for si in range(3):
                e=np.zeros((1,60,10),np.float32)
                if ei>=0:e[:,:,ei]=1
                s=np.zeros((1,60,3),np.float32);s[:,:,si]=1
                inputs=dict(pcm=pcm[:16000][None,:].copy(),emotion=e,gaze_style=s)
                with torch.inference_mode():expected=native(*(torch.from_numpy(inputs[n]) for n in ['pcm','emotion','gaze_style']))
                actual=session.run(None,inputs);err=max(float(np.abs(p-q.numpy()).max()) for p,q in zip(actual,expected))
                parity.append(dict(sample=sample['id'],emotion_index=ei,style=si,error=err))
                if err>3e-5 or any(not np.isfinite(p).all() or p.min()<0 or p.max()>1 for p in actual):failures.append('ONNX parity/bounds')
    for emotion in ['neutral','anger']:
        pcm=np.zeros(48000,np.float32);a=sequence(CURRENT/'model.onnx',pcm,emotion)['blendshapes'];b=sequence(model,pcm,emotion)['blendshapes'];item=dict(emotion=emotion,regions={})
        for name in ['mouth','chin']:
            sa=rms(np.diff(project(a,shape[:,regions[name]])[30:],axis=0));sb=rms(np.diff(project(b,shape[:,regions[name]])[30:],axis=0));item['regions'][name]=dict(before_step_mm=sa,after_step_mm=sb,ratio=sb/sa)
            if sb>sa*.5:failures.append(emotion+' zero-input '+name+' reduction <50%')
        silence.append(item)
    pooled=float(np.sqrt(np.mean([r['regions']['chin']['after_step_mm']**2 for r in rows])/np.mean([r['regions']['chin']['before_step_mm']**2 for r in rows])))
    if pooled>.5:failures.append('Pooled final200ms chin reduction <50%')
    protected=json.loads((OUT/'protected-before.json').read_text())
    historical_drift=[p for p,h in protected.items() if not Path(p).exists() or sha(p)!=h]
    observed=json.loads((APP/'.local/speech-rest-protected-observed.json').read_text())
    if any((sha(p) if Path(p).exists() else None)!=h for p,h in observed.items()):failures.append('Protected source changed since current observation')
    report=dict(model_sha256=sha(model),parent_sha256=sha(CURRENT/'model.onnx'),bytes=model.stat().st_size,passed=not(failures or known),
        rest_checks_passed=not failures,mouth_tail_checks_passed=not failures,violations=known+failures,mouth_tail_failures=failures,
        lip_tolerance=1.05,lip_numerical_tolerance_mm=.0001,phrases=rows,silence=silence,parity=parity,pooled_chin_tail_ratio=pooled,
        protected_primary_files_checked=len(protected),historical_external_source_drift=historical_drift,protected_current_observation_sha256=sha(APP/'.local/speech-rest-protected-observed.json'),review_warnings=known+failures,
        review_notes=['Learned energy-aware rest correction; no runtime silence threshold/end timer.','25 non-mouth controls and articulation preserved.','Known recordings are development checks, not independent release evidence.'])
    shutil.copyfile(CURRENT/'raw-gap-references.npz',DEST/'raw-gap-references.npz');np.savez_compressed(DEST/'rest-traces.npz',**traces)
    (DEST/'verification.json').write_text(json.dumps(report,indent=2));print('CHECKS',not failures,'POOLED',pooled,'FAILURES',failures,'LEGACY',known,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path);parser.add_argument('--kind',choices=['learned','conditioned','weighted','adaptive'],default='learned');args=parser.parse_args();main(args.directory,args.kind)

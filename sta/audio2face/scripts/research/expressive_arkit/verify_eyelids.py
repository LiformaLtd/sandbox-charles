"""Check lid fidelity/tempo, unchanged other channels, lip RMS and ONNX export."""
import json
import numpy as np
import torch
import onnxruntime as ort
from common import *
from eyelid_model import StableLids,NaturalWaveform,LIDS
from train_eyelids import DEST,PARENT,DATA
from infer_full import sequence


def main():
    torch.set_num_threads(4)
    with np.load(DEST/'statistics.npz') as z:net=StableLids(PARENT,z['means'],z['scales'])
    net.load_state_dict(torch.load(DEST/'face.pt',map_location='cpu',weights_only=True));net.eval();native=NaturalWaveform(net).eval()
    with np.load(OUT/'calibration.npz') as z:basis=z['gap_basis'];rest=z['gap_neutral']
    if np.abs(basis[LIDS]).max()!=0:raise ValueError('Lid shapes affect lip-gap basis')
    diagnostic=json.loads((DEST/'diagnostic/metrics.json').read_text())
    if diagnostic['candidate_sha256']!=sha(DEST/'model.onnx'):raise ValueError('Diagnostic SHA mismatch')
    with np.load(DATA/'test.npz') as z:data={k:z[k].copy() for k in ['hidden','baseline','target','valid','emotion']}
    errors={name:np.zeros(6,np.float64) for name in ['parent','new']};count=0
    with torch.inference_mode():
        for start in range(0,len(data['hidden']),128):
            ix=slice(start,start+128);h=torch.from_numpy(data['hidden'][ix]);b=torch.from_numpy(data['baseline'][ix]);e=torch.from_numpy(data['emotion'][ix,None,:]).expand(-1,60,-1)
            style=torch.zeros(len(h),60,3);style[:,:,1]=1
            for name,p in [('parent',net.parent(h,b,e,style)[:,15:45,LIDS]),('new',net.lids(h,e)[:,15:45,LIDS])]:
                error=p.numpy()-data['target'][ix][:,:,LIDS];v=data['valid'][ix];errors[name]+=np.square(error[v]).sum(0)
            count+=data['valid'][ix].sum()
    fidelity={n:np.sqrt(v/count).tolist() for n,v in errors.items()}
    failures=[]
    if errors['new'].sum()>errors['parent'].sum():failures.append('Held-out lid coefficient error worsened')
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(DEST/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
    parent_report=json.loads((PARENT/'verification.json').read_text());refs=np.load(PARENT/'raw-gap-references.npz')
    phrases=[];parity=[];keep=[c for c in range(52) if c not in LIDS]
    for sample in manifest['samples']:
        pcm=np.fromfile(APP/'public'/sample['pcm'].lstrip('/'),'<f4')
        for emotion in ['neutral','anger']:
            before=sequence(PARENT/'model.onnx',pcm,emotion,1.,'James');after=sequence(DEST/'model.onnx',pcm,emotion,1.,'James')
            drift=float(np.abs(before['blendshapes'][:,keep]-after['blendshapes'][:,keep]).max())
            if drift>3e-5:failures.append('Non-lid prediction changed')
            if np.abs(before['articulation']-after['articulation']).max()>3e-5:failures.append('Speech reference changed')
            gb=np.einsum('fc,ck->fk',before['blendshapes'],basis,optimize=False)+rest
            ga=np.einsum('fc,ck->fk',after['blendshapes'],basis,optimize=False)+rest
            if np.abs(ga-gb).max()>3e-5:failures.append('Lip gaps changed')
            key=sample['id']+'--'+emotion;truth=refs[key]
            row=next(r.copy() for r in parent_report['phrases'] if r['sample']==sample['id'] and r['emotion']==emotion)
            row['new_lip_rms_mm']=float(np.sqrt(np.mean((ga-truth)**2)))*10
            row['lip_ratio']=row['new_lip_rms_mm']/row['old_lip_rms_mm'];row['non_lid_max_drift']=drift
            diag=next(r for r in diagnostic['cases'] if r['sample']==sample['id'] and r['emotion']==emotion)
            old=diag['full_surface']['previous_candidate']['fast_4_15_hz_rms'];new=diag['full_surface']['new_candidate']['fast_4_15_hz_rms']
            row['eyelid_fast_rms_mm_before']=old;row['eyelid_fast_rms_mm_after']=new
            if new>.6*old:failures.append(sample['id']+' '+emotion+' insufficient lid jitter reduction')
            phrases.append(row)
        for ei in [-1,1,6,0]:
            for si in range(3):
                e=np.zeros((1,60,10),np.float32)
                if ei>=0:e[:,:,ei]=1
                style=np.zeros((1,60,3),np.float32);style[:,:,si]=1
                inputs=dict(pcm=pcm[:16000][None,:].copy(),emotion=e,gaze_style=style)
                with torch.inference_mode():expected=native(*(torch.from_numpy(inputs[n]) for n in ['pcm','emotion','gaze_style']))
                actual=session.run(None,inputs);error=max(float(np.abs(a-b.numpy()).max()) for a,b in zip(actual,expected))
                if error>3e-5 or any(not np.isfinite(a).all() or a.min()<0 or a.max()>1 for a in actual):failures.append('ONNX parity/validity')
                parity.append(dict(sample=sample['id'],emotion_index=ei,gaze_style=si,max_error=error))
    protected=json.loads((OUT/'protected-before.json').read_text())
    if any(sha(p)!=h for p,h in protected.items()):failures.append('Protected primary files changed')
    known=[r['sample']+' '+r['emotion']+' lip RMS >5%' for r in phrases if r['lip_ratio']>1.05]
    result=dict(model_sha256=sha(DEST/'model.onnx'),bytes=(DEST/'model.onnx').stat().st_size,parent_sha256=sha(PARENT/'model.onnx'),
        passed=not(failures or known),eyelid_checks_passed=not failures,violations=known+failures,eyelid_failures=failures,
        lip_tolerance=1.05,phrases=phrases,parity=parity,held_out_lid_coefficient_rms=fidelity,protected_primary_files_checked=len(protected),
        limitations=['Inherited +5.5% appointment/full-anger lip RMS result remains a visual-review exception.',
          'Teacher labels contain partial blink coefficients; complete spontaneous-blink timing has not been established.',
          'No Player or 2.5D changes.'])
    (DEST/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if failures:raise ValueError('Lid checks failed: '+str(failures))

if __name__=='__main__':main()

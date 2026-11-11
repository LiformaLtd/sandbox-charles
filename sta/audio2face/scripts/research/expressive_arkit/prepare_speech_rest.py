"""Quiet-rest geometry targets from the teacher; no runtime silence/end detector."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import *
from speech_rest_model import DEST,SpeechRestFace,CURRENT
from refine_mouth_tails import NaturalWaveform
from fit import PAIRS,mm

DATA=APP/'.local/expressive-full-neutral-v1'

def end_blend(pcm,n):
    rms=np.array([np.sqrt(np.mean(pcm[i:i+320]**2)) for i in range(0,len(pcm),320)])
    active=np.flatnonzero(rms>max(10**(-55/20),rms.max()*10**(-35/20)))
    endpoint=(int(active[-1])+1)*.02 if len(active) else 0
    # Keep 150 ms of release, then settle to the teacher's own rest geometry.
    alpha=np.clip((np.arange(n)/60-endpoint-.15)/.05,0,1);return alpha*alpha*(3-2*alpha)

def main():
    torch.set_num_threads(4);DEST.mkdir(exist_ok=True)
    if (DEST/'data-protocol.json').exists():raise ValueError('Prepared rest dataset exists')
    with np.load(OUT/'calibration.npz') as z:shape=z['shape'];neutral=z['neutral'];gap=z['gap_basis']*10;rest=z['gap_neutral']*10
    x,y,z=neutral.T;regions=dict(chin=np.flatnonzero((np.abs(x)<2)&(y>162)&(y<166)&(z>3)),mouth=np.flatnonzero((np.abs(x)<3)&(y>165.8)&(y<170.2)&(z>4.5)))
    embeddings=dict(corner=shape[:,PAIRS[[0,-1]]].mean(2).reshape(52,6)*10);projectors={}
    for name,roi in regions.items():
        basis=shape[:,roi].reshape(52,-1).astype(np.float64)*10/np.sqrt(len(roi)*3)
        u,s,v=np.linalg.svd(basis,full_matrices=False);keep=s>.0001;k=int(keep.sum())
        embeddings[name]=(u[:,keep]*s[keep]*np.sqrt(k)).astype(np.float32)
        projectors[name]=(v[keep].T*10/np.sqrt(len(roi)*3)*np.sqrt(k)).astype(np.float32)
    upper=np.unique(np.concatenate([np.argsort(np.max(np.linalg.norm(shape[41:46],axis=2),axis=0))[-512:],*[np.argsort(np.linalg.norm(shape[c],axis=1))[-256:] for c in [0,5,6,7,12,13]]]))
    b=shape[:,upper].reshape(52,-1).astype(np.float64)*10/np.sqrt(len(upper)*3);u,s,_=np.linalg.svd(b,full_matrices=False);embeddings['upper']=(u*s*np.sqrt(52)).astype(np.float32)
    np.savez(DEST/'geometry.npz',**embeddings,gap=gap,rest=rest)
    rows={r['stem']:r for r in json.loads((OUT/'manifest.json').read_text())};cache={};quiet_targets={name:[] for name in [*regions,'corner','gaps']}
    for split in ['train','dev','test']:
        with np.load(DATA/(split+'.npz')) as z:d={k:z[k].copy() for k in ['hidden','baseline','emotion','valid','voices','cases']}
        targets={name:np.zeros((len(d['hidden']),30,embedding.shape[1]),np.float32) for name,embedding in embeddings.items() if name!='upper'}
        targets['gaps']=np.zeros((len(d['hidden']),30,9),np.float32);raw_gaps=np.zeros_like(targets['gaps']);cursor={}
        for i,case in enumerate(d['cases']):
            case=str(case)
            if case not in cache:
                stem,emotion=case.rsplit('--',1);row=rows[stem];pcm=np.fromfile(row['audio'],'<f4')
                if sha(row['audio'])!=row['sha256_audio']:raise ValueError('Changed source audio')
                with np.load(DATA/'labels'/f'{case}.npz') as z:w=z['weights'];g=z['gaps']*10
                alpha=end_blend(pcm,len(w))[:,None]
                raw={name:mm(w,embedding) for name,embedding in embeddings.items() if name!='upper'};raw['gaps']=g
                cleaned={name:(values*(1-alpha)+values[-9:].mean(0,keepdims=True)*alpha).astype(np.float32) for name,values in raw.items()}
                cache[case]=(cleaned,g)
                if split=='train':
                    condition=0 if emotion=='neutral' else EMOTIONS.index(emotion)+1
                    for name in quiet_targets:quiet_targets[name].append((condition,raw[name][-9:].mean(0)))
            if case not in cursor:cursor[case]=0
            v=d['valid'][i];count=int(v.sum());start=cursor[case]
            for name in targets:targets[name][i,v]=cache[case][0][name][start:start+count]
            raw_gaps[i,v]=cache[case][1][start:start+count];cursor[case]+=count
        np.savez_compressed(DEST/(split+'.npz'),**d,**{'target_'+k:v for k,v in targets.items()},raw_gaps=raw_gaps)
        print('rest data',split,len(d['hidden']),flush=True)
    # The explicitly trained appointment/anger development regression.
    with np.load(APP/'.local/corner-appointment-regression/features.npz') as z:d={k:z[k].copy() for k in ['hidden','baseline','emotion','valid','voices','cases']}
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text());ref=manifest['teacherReviews']['appointment--anger'];path=APP/'public'/ref['url'].lstrip('/')
    if sha(path)!=ref['sha256']:raise ValueError('Teacher replay changed')
    raw=np.fromfile(path,'<f4').reshape(ref['frames'],88831)
    with np.load(ASSETS/'model_data_James.npz') as z:skin=raw[:,:72006].reshape(-1,24002,3)+z['neutral_skin']-neutral
    pcm=np.fromfile(APP/'public'/next(s['pcm'] for s in manifest['samples'] if s['id']=='appointment').lstrip('/'),'<f4');alpha=end_blend(pcm,len(raw))[:,None]
    truth={name:mm(skin[:,roi].reshape(len(raw),-1),projectors[name]) for name,roi in regions.items()}
    truth['corner']=skin[:,PAIRS[[0,-1]]].mean(2).reshape(len(raw),6)*10
    absolute=skin+neutral;truth['gaps']=(absolute[:,PAIRS[:,0],1]-absolute[:,PAIRS[:,1],1])*10
    targets={name:(values*(1-alpha)+values[-9:].mean(0,keepdims=True)*alpha).astype(np.float32) for name,values in truth.items()}
    cursor=0;windowed={name:np.zeros((len(d['hidden']),30,v.shape[1]),np.float32) for name,v in truth.items()};gaps_raw=np.zeros_like(windowed['gaps'])
    for i,v in enumerate(d['valid']):
        count=int(v.sum())
        for name in windowed:windowed[name][i,v]=targets[name][cursor:cursor+count]
        gaps_raw[i,v]=truth['gaps'][cursor:cursor+count];cursor+=count
    np.savez_compressed(DEST/'regression.npz',**d,**{'target_'+k:v for k,v in windowed.items()},raw_gaps=gaps_raw)
    # Quiet examples teach a stationary teacher-derived rest, including anger.
    wave=NaturalWaveform(SpeechRestFace()).speech.eval();student=wave.student;rng=np.random.default_rng(601);hs=[];bs=[]
    with torch.inference_mode():
        for amplitude in [0.,1e-6,1e-5,3e-5,1e-4]:
            for _ in range(3):
                pcm=rng.normal(0,amplitude,(1,16000)).astype(np.float32);spectrum=F.conv1d(torch.from_numpy(pcm).unsqueeze(1),wave.filters,stride=160,padding=200);real,imag=spectrum.chunk(2,dim=1)
                mel=(1+F.conv1d(real.square()+imag.square(),wave.mel)*10).log();h=student.blocks(F.gelu(student.encoder(mel),approximate='tanh'))
                aligned=h.index_select(2,student.left)+student.fraction*(h.index_select(2,student.left+1)-h.index_select(2,student.left))
                speech=(student.output(aligned).transpose(1,2)*student.scale+student.mean).clamp(0,1)@student.channel_map
                hs.extend([aligned.transpose(1,2)[0].numpy()]*11);bs.extend([speech[0].numpy()]*11)
    n=len(hs);emotion=np.zeros((n,10),np.float32)
    for i in range(n):
        if i%11:emotion[i,i%11-1]=1
    q=dict(hidden=np.stack(hs),baseline=np.stack(bs),emotion=emotion,valid=np.ones((n,30),bool),voices=np.array(['quiet']*n),cases=np.array([f'quiet-{i}' for i in range(n)]))
    for name,values in quiet_targets.items():
        rest_targets=np.stack([np.mean([v for e,v in values if e==condition],axis=0) for condition in range(11)])
        q['target_'+name]=np.repeat(rest_targets[np.arange(n)%11,None,:],30,axis=1).astype(np.float32)
    q['raw_gaps']=q['target_gaps'].copy();np.savez_compressed(DEST/'quiet.npz',**q)
    freeze(DEST/'data-protocol.json',dict(source_sha256=sha(__file__),parent_sha256=sha(CURRENT/'model.onnx'),
        source_features={s:sha(DATA/(s+'.npz')) for s in ['train','dev','test']},
        label_policy='Teacher-fitted geometry; after 150ms trailing acoustic silence, 50ms smooth transition to teacher final150ms mean. Raw gaps retained separately for acceptance.',
        quiet='Zero and -120..-80 dBFS noise; constant teacher-derived rest per supplied emotion, training-only means',
        regression='Appointment/anger raw teacher geometry, explicitly used for development',regions={k:len(v) for k,v in regions.items()}))
    print('quiet data',n,flush=True)

if __name__=='__main__':main()

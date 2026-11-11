"""Explicit appointment/anger development regression; not held-out evaluation."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import *
from guard_mouth_tails import GapGuardFace
from refine_mouth_tails import NaturalWaveform
from fit import PAIRS


def main():
    torch.set_num_threads(4)
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text());sample=next(s for s in manifest['samples'] if s['id']=='appointment')
    source=APP/'public'/sample['pcm'].lstrip('/');pcm=np.fromfile(source,'<f4')
    if sha(source)!=sample['sha256']:raise ValueError('Changed appointment audio')
    ref=manifest['teacherReviews']['appointment--anger'];track=APP/'public'/ref['url'].lstrip('/')
    if sha(track)!=ref['sha256'] or ref['audioSha256']!=sample['sha256']:raise ValueError('Teacher regression mismatch')
    teacher=np.fromfile(track,'<f4').reshape(ref['frames'],88831)
    with np.load(OUT/'calibration.npz') as z:neutral=z['neutral'];shape=z['shape']
    with np.load(ASSETS/'model_data_James.npz') as z:teacher_neutral=z['neutral_skin']
    skin=teacher[:,:72006].reshape(-1,24002,3)+teacher_neutral
    corners=(skin[:,PAIRS[[0,-1]]].mean(2)-neutral[PAIRS[[0,-1]]].mean(1)).reshape(-1,6)
    gaps=skin[:,PAIRS[:,0],1]-skin[:,PAIRS[:,1],1]
    wave=NaturalWaveform(GapGuardFace()).speech.eval();student=wave.student;hs=[];bs=[];ys=[];gs=[];vs=[]
    with torch.inference_mode():
        for start in range(-8000,len(pcm)-4000,8000):
            window=np.zeros((1,16000),np.float32);lo,hi=max(0,start),min(len(pcm),start+16000);window[0,lo-start:hi-start]=pcm[lo:hi]
            spectrum=F.conv1d(torch.from_numpy(window).unsqueeze(1),wave.filters,stride=160,padding=200);real,imag=spectrum.chunk(2,dim=1)
            mel=(1+F.conv1d(real.square()+imag.square(),wave.mel)*10).log();h=student.blocks(F.gelu(student.encoder(mel),approximate='tanh'))
            aligned=h.index_select(2,student.left)+student.fraction*(h.index_select(2,student.left+1)-h.index_select(2,student.left))
            speech=(student.output(aligned).transpose(1,2)*student.scale+student.mean).clamp(0,1)@student.channel_map
            frame=start*60//16000+np.arange(15,45);valid=(frame>=0)&(frame<len(corners));y=np.zeros((30,6),np.float32);g=np.zeros((30,9),np.float32)
            y[valid]=corners[frame[valid]];g[valid]=gaps[frame[valid]]
            hs.append(aligned.transpose(1,2)[0].numpy());bs.append(speech[0].numpy());ys.append(y);gs.append(g);vs.append(valid)
    n=len(hs);valid=np.stack(vs);following=np.minimum(np.arange(n)+1,n-1);overlap=np.concatenate([valid[:,25:30],valid[following,:5]],axis=1);overlap[-1]=False
    tail=np.zeros_like(valid);cursor=0
    for i,v in enumerate(valid):
        count=int(v.sum());tail[i,v]=np.arange(cursor,cursor+count)>=len(corners)-48;cursor+=count
    assert cursor==ref['frames']
    emotion=np.zeros((n,10),np.float32);emotion[:,EMOTIONS.index('anger')]=1
    out=APP/'.local/corner-appointment-regression';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'features.npz',hidden=np.stack(hs),baseline=np.stack(bs),target=np.zeros((n,30,52),np.float32),
        corner_target=np.stack(ys),gaps=np.stack(gs),valid=valid,following=following,overlap=overlap,tail=tail,emotion=emotion,
        voices=np.array(['supertonic:M1-regression']*n),cases=np.array(['appointment--anger']*n))
    freeze(out/'provenance.json',dict(audio_sha256=sha(source),teacher_geometry_sha256=sha(track),teacher_model_sha256=ref['teacherModelSha256'],
        source_sha256=sha(__file__),purpose='Explicit development regression trained on request; not independent confirmation',frames=cursor,windows=n))
    print('Prepared regression',n,'windows',cursor,'frames',flush=True)

if __name__=='__main__':main()

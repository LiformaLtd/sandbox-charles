"""Explicit development anchors for all five reviewed phrases; not held-out tests."""
import json
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from common import *
from speech_rest_energy_model import EnergyRestFace,EnergyRestWaveform
from prepare_speech_rest import end_blend

DEST=APP/'.local/speech-rest-v4'

def main():
    DEST.mkdir(exist_ok=True);torch.set_num_threads(4)
    source=APP/'.local/speech-rest-v3'
    for n in ['train.npz','dev.npz','quiet.npz','geometry.npz','train-base.npy','dev-base.npy','quiet-base.npy']:shutil.copyfile(source/n,DEST/n)
    with np.load(DEST/'geometry.npz') as z:geo={k:torch.from_numpy(z[k]) for k in z.files}
    model=EnergyRestWaveform(EnergyRestFace()).eval();speech=model.speech;s=speech.student
    refs=np.load(APP/'.local/mouth-corner-isolated-v1/raw-gap-references.npz');corners=np.load(APP/'.local/mouth-tail-diagnostic/traces.npz')
    fields={n:[] for n in ['hidden','baseline','emotion','energy','valid','voices','cases','target_corner','target_chin','target_mouth','target_gaps','raw_gaps','quiet_mask']};bases=[]
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
    with torch.inference_mode():
        for sample in manifest['samples']:
            pcm=np.fromfile(APP/'public'/sample['pcm'].lstrip('/'),'<f4');count=(len(pcm)*60+15999)//16000;alpha=end_blend(pcm,count)
            for emotion in ['neutral','anger']:
                key=sample['id']+'--'+emotion;e=torch.zeros(1,60,10)
                if emotion!='neutral':e[:,:,EMOTIONS.index(emotion)]=1
                style=torch.zeros(1,60,3);style[:,:,1]=1;tracks=[];windows=[]
                for start in range(-8000,len(pcm)-4000,8000):
                    x=np.zeros((1,16000),np.float32);lo,hi=max(0,start),min(len(pcm),start+16000);x[0,lo-start:hi-start]=pcm[lo:hi];t=torch.from_numpy(x)
                    spec=F.conv1d(t.unsqueeze(1),speech.filters,stride=160,padding=200);real,imag=spec.chunk(2,1);mel=(1+F.conv1d(real.square()+imag.square(),speech.mel)*10).log();h=s.blocks(F.gelu(s.encoder(mel),approximate='tanh'));aligned=h.index_select(2,s.left)+s.fraction*(h.index_select(2,s.left+1)-h.index_select(2,s.left));hidden=aligned.transpose(1,2);b=(s.output(aligned).transpose(1,2)*s.scale+s.mean).clamp(0,1)@s.channel_map
                    base=model.face.base(hidden,b,e,style)[0].numpy();positions=start+np.arange(15,45)*16000//60;valid=(positions>=0)&(positions<len(pcm));tracks.append(base[15:45][valid]);windows.append((hidden[0].numpy(),b[0].numpy(),model.energy(t)[0].numpy(),valid,base))
                full=np.concatenate(tracks);targets={}
                for name in ['chin','mouth']:
                    values=torch.from_numpy(full)@geo[name];v=values.numpy();targets[name]=v*(1-alpha[:,None])+v[-9:].mean(0)*alpha[:,None]
                targets['corner']=corners[key+'--teacher_raw--points'].reshape(count,6);targets['gaps']=refs[key]*10
                cursor=0
                for hidden,b,energy,valid,base in windows:
                    n=valid.sum();fields['hidden'].append(hidden);fields['baseline'].append(b);fields['energy'].append(energy);fields['emotion'].append(e[0,0].numpy());fields['valid'].append(valid);fields['voices'].append('review-regression');fields['cases'].append(key);bases.append(base)
                    quiet=np.zeros(30,bool);quiet[valid]=alpha[cursor:cursor+n]>.99;fields['quiet_mask'].append(quiet)
                    for name in ['corner','chin','mouth','gaps']:
                        window=np.zeros((30,targets[name].shape[1]),np.float32);window[valid]=targets[name][cursor:cursor+n];fields['target_'+name].append(window)
                    fields['raw_gaps'].append(fields['target_gaps'][-1]);cursor+=n
    np.savez_compressed(DEST/'regression.npz',**{k:np.array(v) for k,v in fields.items()});np.save(DEST/'regression-base.npy',np.array(bases))
    freeze(DEST/'regression-protocol.json',dict(source=sha(__file__),phrase_policy='All five review phrases neutral/anger are explicit development anchors. Not independent tests.',targets='Frozen-current active chin/mouth; settled own mean during labelled trailing quiet; raw teacher corners/gaps throughout.'))
    print('regression windows',len(bases))

if __name__=='__main__':main()

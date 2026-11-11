"""Probe continuous audio-energy conditioning, with stationary emotion rest poses."""
import json
import numpy as np
import torch
from common import *
from infer_full import sequence
from speech_rest_energy_model import EnergyRestFace,EnergyRestWaveform,CURRENT
from fit import PAIRS

def main():
 torch.set_num_threads(4);wave=EnergyRestWaveform(EnergyRestFace()).eval()
 with np.load(OUT/'calibration.npz') as z:shape=z['shape'];neutral=z['neutral'];gap=z['gap_basis'];rest=z['gap_neutral']
 x,y,z=neutral.T;roi=np.flatnonzero((np.abs(x)<2)&(y>162)&(y<166)&(z>3));corn=shape[:,PAIRS[[0,-1]]].mean(2)
 refs=np.load(CURRENT/'raw-gap-references.npz');truth=np.load(APP/'.local/mouth-tail-diagnostic/traces.npz');manifest=json.loads((APP/'public/assets/expressive/review.json').read_text());rows=[]
 g=gap[14:41].astype(np.float64);u,ss,_=np.linalg.svd(g,full_matrices=False);u=u[:,ss>ss[0]*1e-8];projection=np.einsum('ik,jk->ij',u,u,optimize=False).astype(np.float32)
 for sample in manifest['samples']:
  pcm=np.fromfile(APP/'public'/sample['pcm'].lstrip('/'),'<f4');energy=[]
  with torch.inference_mode():
   for start in range(-8000,len(pcm)-4000,8000):
    xx=np.zeros((1,16000),np.float32);lo,hi=max(0,start),min(len(pcm),start+16000);xx[0,lo-start:hi-start]=pcm[lo:hi];pos=start+np.arange(15,45)*16000//60;energy.append(wave.energy(torch.from_numpy(xx))[0,15:45,1].numpy()[(pos>=0)&(pos<len(pcm))])
  energy=np.concatenate(energy)
  for emotion in ['neutral','anger']:
   key=sample['id']+'--'+emotion;a=sequence(CURRENT/'model.onnx',pcm,emotion)['blendshapes'];zero=sequence(CURRENT/'model.onnx',np.zeros(32000,np.float32),emotion)['blendshapes'][30:60,14:41].mean(0)
   def rms(v):return float(np.sqrt(np.mean(v**2)))
   def chin(w):return np.einsum('fc,cvk->fvk',w[-13:],shape[:,roi],optimize=False)*10
   def gaps(w):return (np.einsum('fc,ck->fk',w,gap,optimize=False)+rest-refs[key])*10
   def corners(w):return np.einsum('fc,cvk->fvk',w,corn,optimize=False)*10-truth[key+'--teacher_raw--points']
   for k in [.03,.1,.3]:
    b=a.copy();activity=energy/(energy+k);b[:,14:41]=a[:,14:41]*activity[:,None]+zero*(1-activity[:,None])
    desired=torch.from_numpy(b[:,14:41]);ref=torch.from_numpy(a[:,14:41]);pr=torch.from_numpy(projection)
    with torch.inference_mode():b[:,14:41]=wave.face.base.project(desired,ref,pr,256).numpy()
    row=dict(sample=sample['id'],emotion=emotion,k=k,chin_ratio=rms(np.diff(chin(b),axis=0))/rms(np.diff(chin(a),axis=0)),gap_ratio=rms(gaps(b))/rms(gaps(a)),corner_ratio=rms(corners(b)[-48:])/rms(corners(a)[-48:]))
    rows.append(row);print(row,flush=True)
 (APP/'.local/quiet-gap-probe.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()

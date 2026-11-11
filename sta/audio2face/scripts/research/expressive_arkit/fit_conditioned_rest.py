"""Fit stationary expressive rest poses; all reviewed phrases are development."""
import json
import time
import numpy as np
import torch
from common import *
from conditioned_rest_model import ConditionedRestWaveform,DEST,CURRENT
from infer_full import sequence
from fit import PAIRS

def main():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);model=ConditionedRestWaveform();manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
 with np.load(OUT/'calibration.npz') as z:gap=torch.from_numpy(z['gap_basis']*10);rest=torch.from_numpy(z['gap_neutral']*10);corner=torch.from_numpy(z['shape'][:,PAIRS[[0,-1]]].mean(2).reshape(52,6)*10)
 with np.load(APP/'.local/speech-rest-v3/geometry.npz') as z:geo={k:torch.from_numpy(z[k]) for k in z.files}
 refs=np.load(CURRENT/'raw-gap-references.npz');corners=np.load(APP/'.local/mouth-tail-diagnostic/traces.npz');cases=[]
 # Teacher-fitted quiet targets cover all eleven controls, not just review anger.
 with np.load(APP/'.local/speech-rest-v1/quiet.npz') as z:q={k:torch.from_numpy(z[k]) for k in z.files if k.startswith('target_')}
 quietbase=np.load(APP/'.local/speech-rest-v3/quiet-base.npy');qb=torch.from_numpy(quietbase[:11,15:45].mean(1));model.rest.data.copy_(qb[:,14:41])
 for sample in manifest['samples']:
  pcm=np.fromfile(APP/'public'/sample['pcm'].lstrip('/'),'<f4');energy=[]
  with torch.inference_mode():
   for start in range(-8000,len(pcm)-4000,8000):
    xx=np.zeros((1,16000),np.float32);lo,hi=max(0,start),min(len(pcm),start+16000);xx[0,lo-start:hi-start]=pcm[lo:hi];pos=start+np.arange(15,45)*16000//60;energy.append(model.activity(torch.from_numpy(xx))[0,15:45][(pos>=0)&(pos<len(pcm))])
  activity=torch.cat(energy)
  for ei,emotion in [(0,'neutral'),(2,'anger')]:
   key=sample['id']+'--'+emotion;base=torch.from_numpy(sequence(CURRENT/'model.onnx',pcm,emotion)['blendshapes']);truth=torch.from_numpy(refs[key]*10);ctruth=torch.from_numpy(corners[key+'--teacher_raw--points'].reshape(len(base),6));ge=((base@gap+rest-truth)**2).mean();ce=((base[-48:]@corner-ctruth[-48:])**2).mean()
   cases.append((ei,base,activity,truth,ctruth,ge,ce,key))
 opt=torch.optim.Adam([model.rest],lr=.003);best=None;score=float('inf');history=[]
 for step in range(2001):
  loss=torch.tensor(0.);ratios=[];cratios=[]
  for ei,base,activity,truth,ctruth,ge,ce,key in cases:
   mouth=base[:,14:41]*activity+model.rest[ei].clamp(0,1)*(1-activity);p=torch.cat([base[:,:14],mouth,base[:,41:]],-1)
   g=((p@gap+rest-truth)**2).mean();c=((p[-48:]@corner-ctruth[-48:])**2).mean();ratios.append(float((g/ge).sqrt().detach()));cratios.append(float((c/ce).sqrt().detach()))
   loss=loss+g+c+200*torch.relu(g-ge*1.03**2)+200*torch.relu(c-ce*1.03**2)
  full=torch.cat([qb[:,:14],model.rest.clamp(0,1),qb[:,41:]],-1)
  for name in ['corner','chin','mouth']:
   target=q['target_'+name][:11].mean(1);loss=loss+.01*((full@geo[name]-target)**2).mean()
  if max(ratios)<=1.05 and max(cratios)<=1.05 and float(loss.detach())<score:best=model.rest.detach().clone();score=float(loss.detach());best_step=step
  if step%200==0:
   history.append(dict(step=step,gap=max(ratios),corner=max(cratios),loss=float(loss.detach())));print(history[-1],flush=True)
  opt.zero_grad();loss.backward();opt.step();model.rest.data.clamp_(0,1)
 if best is None:raise ValueError('No stationary rest fit preserves phrase lip/corner checks')
 model.rest.data.copy_(best);torch.save(model.state_dict(),DEST/'wave.pt');freeze(DEST/'fit.json',dict(best_step=best_step,history=history,source_sha256=sha(__file__),model_source_sha256=sha(Path(__file__).with_name('conditioned_rest_model.py')),parent=sha(CURRENT/'model.onnx'),development='All five review phrases neutral/anger used to fit/guard rest poses; not independent validation',activity='1-exp(-continuous400ms-log-power/.1); no binary VAD or clip-end timer'))
 print('SELECTED',best_step,flush=True)
if __name__=='__main__':main()

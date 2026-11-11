"""Fit stationary expressive rest poses; all reviewed phrases are development."""
import json
import time
import numpy as np
import torch
from common import *
from adaptive_rest_model import AdaptiveRestWaveform as ConditionedRestWaveform
DEST=APP/'.local/adaptive-rest-v1'
from speech_rest_energy_model import CURRENT
import onnxruntime as ort
from infer_full import sequence
from fit import PAIRS

def main():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);model=ConditionedRestWaveform();manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
 with np.load(OUT/'calibration.npz') as z:gap=torch.from_numpy(z['gap_basis']*10);rest=torch.from_numpy(z['gap_neutral']*10);corner=torch.from_numpy(z['shape'][:,PAIRS[[0,-1]]].mean(2).reshape(52,6)*10)
 with np.load(APP/'.local/speech-rest-v3/geometry.npz') as z:geo={k:torch.from_numpy(z[k]) for k in z.files}
 refs=np.load(CURRENT/'raw-gap-references.npz');corners=np.load(APP/'.local/mouth-tail-diagnostic/traces.npz');cases=[]
 # Teacher-fitted quiet targets cover all eleven controls, not just review anger.
 with np.load(APP/'.local/speech-rest-v1/quiet.npz') as z:q={k:torch.from_numpy(z[k]) for k in z.files if k.startswith('target_')}
 quietbase=np.load(APP/'.local/speech-rest-v3/quiet-base.npy');qb=torch.from_numpy(quietbase[:11,15:45].mean(1));model.rest.data.zero_()
 options=ort.SessionOptions();options.intra_op_num_threads=4;session=ort.InferenceSession(str(CURRENT/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
 for sample in manifest['samples']:
  pcm=np.fromfile(APP/'public'/sample['pcm'].lstrip('/'),'<f4');energy=[]
  with torch.inference_mode():
   for start in range(-8000,len(pcm)-4000,8000):
    xx=np.zeros((1,16000),np.float32);lo,hi=max(0,start),min(len(pcm),start+16000);xx[0,lo-start:hi-start]=pcm[lo:hi];pos=start+np.arange(15,45)*16000//60;energy.append(model.activity(torch.from_numpy(xx))[0,15:45][(pos>=0)&(pos<len(pcm))])
  activity=torch.cat(energy)
  for ei,emotion in [(0,'neutral'),(2,'anger')]:
   means=[];e=np.zeros((1,60,10),np.float32);style=np.zeros((1,60,3),np.float32);style[:,:,1]=1
   if ei:e[:,:,EMOTIONS.index(emotion)]=1
   for start in range(-8000,len(pcm)-4000,8000):
    xx=np.zeros((1,16000),np.float32);lo,hi=max(0,start),min(len(pcm),start+16000);xx[0,lo-start:hi-start]=pcm[lo:hi];pos=start+np.arange(15,45)*16000//60;valid=(pos>=0)&(pos<len(pcm))
    out=torch.from_numpy(session.run(['blendshapes'],dict(pcm=xx,emotion=e,gaze_style=style))[0])
    with torch.inference_mode():quiet=1-model.activity(torch.from_numpy(xx));mean=(out[:,:,14:41]*quiet).sum(1)/quiet.sum(1).clamp_min(1e-6);means.append(mean.expand(int(valid.sum()),27).clone())
   mean=torch.cat(means)
   key=sample['id']+'--'+emotion;base=torch.from_numpy(sequence(CURRENT/'model.onnx',pcm,emotion)['blendshapes']);truth=torch.from_numpy(refs[key]*10);ctruth=torch.from_numpy(corners[key+'--teacher_raw--points'].reshape(len(base),6));ge=((base@gap+rest-truth)**2).mean();ce=((base[-48:]@corner-ctruth[-48:])**2).mean()
   cases.append((ei,base,activity,truth,ctruth,ge,ce,key,mean))
 opt=torch.optim.Adam([model.rest,*model.rest_transform.parameters()],lr=.001);best=None;score=float('inf');history=[]
 for step in range(2001):
  loss=torch.tensor(0.);ratios=[];cratios=[]
  for ei,base,activity,truth,ctruth,ge,ce,key,mean in cases:
   mouth=base[:,14:41]*activity+(mean+model.rest_transform(mean)+model.rest[ei]).clamp(0,1)*(1-activity);p=torch.cat([base[:,:14],mouth,base[:,41:]],-1)
   g=((p@gap+rest-truth)**2).mean();c=((p[-48:]@corner-ctruth[-48:])**2).mean();ratios.append(float((g/ge).sqrt().detach()));cratios.append(float((c/ce).sqrt().detach()))
   loss=loss+g+c+200*torch.relu(g-ge*1.03**2)+200*torch.relu(c-ce*1.03**2)
  full=torch.cat([qb[:,:14],(qb[:,14:41]+model.rest+model.rest_transform(qb[:,14:41])).clamp(0,1),qb[:,41:]],-1)
  loss=loss+.01*model.rest_transform.weight.square().mean()
  for name in ['corner','chin','mouth']:
   target=q['target_'+name][:11].mean(1);loss=loss+.01*((full@geo[name]-target)**2).mean()
  if max(ratios)<=1.05 and max(cratios)<=1.05 and float(loss.detach())<score:best={k:v.detach().clone() for k,v in model.state_dict().items()};score=float(loss.detach());best_step=step
  if step%200==0:
   history.append(dict(step=step,gap=max(ratios),corner=max(cratios),loss=float(loss.detach())));print(history[-1],flush=True)
  opt.zero_grad();loss.backward();opt.step();model.rest.data.clamp_(-1,1)
 if best is None:raise ValueError('No stationary rest fit preserves phrase lip/corner checks')
 model.load_state_dict(best);torch.save(model.state_dict(),DEST/'wave.pt');freeze(DEST/'fit.json',dict(best_step=best_step,history=history,source_sha256=sha(__file__),model_source_sha256=sha(Path(__file__).with_name('adaptive_rest_model.py')),parent=sha(CURRENT/'model.onnx'),development='All five review phrases neutral/anger used to fit/guard rest poses; not independent validation',activity='1-exp(-continuous400ms-log-power/.1); no binary VAD or clip-end timer'))
 print('SELECTED',best_step,flush=True)
if __name__=='__main__':main()

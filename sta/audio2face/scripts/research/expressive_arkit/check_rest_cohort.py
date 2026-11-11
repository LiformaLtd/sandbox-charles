"""Check rest adapter on voice windows excluded from its fitting."""
import json
import numpy as np
import torch
from common import *
from adaptive_rest_model import AdaptiveRestWaveform,DEST

def main():
 torch.set_num_threads(4);model=AdaptiveRestWaveform();model.load_rest(DEST/'wave.pt');model.eval()
 source=APP/'.local/speech-rest-v3'
 with np.load(source/'dev.npz') as z:d={k:z[k].copy() for k in ['emotion','cases','valid','raw_gaps']}
 base=np.load(source/'dev-base.npy');rows={r['stem']:r for r in json.loads((OUT/'manifest.json').read_text())};cursors={};cache={};audio={};activities=[]
 with torch.inference_mode():
  for case in d['cases']:
   case=str(case);stem=case.rsplit('--',1)[0];i=cursors.get(case,0);cursors[case]=i+1;key=(stem,i)
   if key not in cache:
    if stem not in audio:
     path=rows[stem]['audio'];audio[stem]=np.fromfile(path,'<f4')
     if sha(path)!=rows[stem]['sha256_audio']:raise ValueError('Source audio changed')
    pcm=audio[stem];start=(i-1)*8000;lo,hi=max(0,start),min(len(pcm),start+16000);x=np.zeros((1,16000),np.float32);x[0,lo-start:hi-start]=pcm[lo:hi];cache[key]=model.activity(torch.from_numpy(x))[0]
   activities.append(cache[key])
  activity=torch.stack(activities);a=torch.from_numpy(base);e=torch.from_numpy(d['emotion'])[:,None,:].expand(-1,60,-1);rest=model.rest_pose(a,activity,e);mouth=a[:,:,14:41]*activity+rest*(1-activity);b=torch.cat([a[:,:,:14],mouth,a[:,:,41:]],-1).numpy()
 with np.load(source/'geometry.npz') as z:g=z['gap'];r=z['rest']
 ga=np.einsum('ntc,ck->ntk',base[:,15:45],g,optimize=False)+r;gb=np.einsum('ntc,ck->ntk',b[:,15:45],g,optimize=False)+r;metrics={}
 for index in [-1,*range(10)]:
  take=(d['emotion'].sum(1)==0) if index==-1 else d['emotion'][:,index]>0
  v=d['valid'][take];old=float(np.sqrt(np.mean((ga[take]-d['raw_gaps'][take])[v]**2)));new=float(np.sqrt(np.mean((gb[take]-d['raw_gaps'][take])[v]**2)));metrics['neutral' if index==-1 else EMOTIONS[index]]=dict(before_rms_mm=old,after_rms_mm=new,ratio=new/old,windows=int(take.sum()))
 report=dict(passed=max(m['ratio'] for m in metrics.values())<=1.05,windows=len(base),controls=metrics,non_mouth_max_drift=float(np.abs(base[:,:,[*range(14),*range(41,52)]]-b[:,:,[*range(14),*range(41,52)]]).max()),scope='627 development voice windows excluded from fitting this adapter; known review phrases separately used for fitting, not independent test')
 (DEST/'cohort-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

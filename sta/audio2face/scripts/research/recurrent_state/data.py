"""Chronological truncated-BPTT streams; immutable existing frontend maps."""
from common import *
import numpy as np
from features import Dataset

def clips(names):
 result=[];cursor=0
 for name in names:
  if name.startswith('replay'):root=WORK/'arkit-student-v3';split=name.split('-')[1]
  elif name=='extra-train':root=WORK/'arkit-student-v3/extra';split='train'
  else:root=SCALE/'datasets'/name;split='dev' if name=='common-dev' else 'train'
  rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']==split]
  for row in rows:
   n=len(range(-8000,Path(row['audio']).stat().st_size//4-4000,8000))
   result.append(dict(start=cursor,count=n,voice=row.get('engine','replay')+':'+row.get('voice','Easton'),stem=row['stem'],name=name));cursor+=n
 assert cursor==Dataset(names).length
 return result

TRAIN=['old-train','new-train','replay-train','extra-train'];DEV=['common-dev','replay-dev']
def batch(dataset,indices):
 n,k=indices.shape;safe=np.maximum(indices,0);b=dataset.batch(safe.ravel());b['valid'] &= (indices.ravel()>=0)[:,None]
 return dict(mel=b['mel'].reshape(n,k,64,101),weights=b['weights'].reshape(n,k*30,39),gaps=b['gaps'].reshape(n,k*30,9),valid=b['valid'].reshape(n,k*30))

def schedule(steps=1240):
 allclips=clips(TRAIN);primary=[i for i,c in enumerate(allclips) if c['name'] in TRAIN[:2]];replay=[i for i,c in enumerate(allclips) if c['name'] in TRAIN[2:]]
 voices=sorted({allclips[i]['voice'] for i in primary});assert len(voices)==8
 groups=[[i for i in primary if allclips[i]['voice']==v] for v in voices for _ in range(2)]+[replay]*4
 rng=np.random.default_rng(1037);chosen=np.full(20,-1);position=np.zeros(20,int);indices=np.full((steps,20,4),-1,np.int32);resets=np.zeros((steps,20),bool)
 for s in range(steps):
  for j,group in enumerate(groups):
   if chosen[j]<0 or position[j]>=allclips[chosen[j]]['count']:chosen[j]=rng.choice(group);position[j]=0;resets[s,j]=True
   c=allclips[chosen[j]];count=min(4,c['count']-position[j]);indices[s,j,:count]=c['start']+np.arange(position[j],position[j]+count);position[j]+=count
 return indices,resets

def prepare():
 OUT.mkdir(exist_ok=True);ix,resets=schedule()
 np.savez(OUT/'schedule.npz',indices=ix,resets=resets)
 freeze(OUT/'clip-map.json',dict(train=clips(TRAIN),dev=clips(DEV)))
 print('Frozen',ix.shape,'chronological indices; schedule sha',sha(OUT/'schedule.npz'),flush=True)
if __name__=='__main__':prepare()

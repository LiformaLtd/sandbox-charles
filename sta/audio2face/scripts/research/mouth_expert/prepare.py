"""Choose cases and verify geometric corruptions without consulting expert scores."""
from common import *
import numpy as np
sys.path.insert(0,str(APP/'scripts/research/multivoice'));from closure_metrics import runs,associate

def variants(weights,gaps,jaw,jaw_basis,closure_mask):
 result={'correct':weights.copy(),'underopen-50':weights.copy(),'missing-closure':weights.copy()}
 median=np.median(gaps,-1);goal=np.where(median>.1,.1+.5*(median-.1),median)
 low=np.zeros(len(weights));high=.5*weights[:,jaw]
 for _ in range(24):
  mid=(low+high)/2;candidate=np.median(gaps-mid[:,None]*jaw_basis,-1);ok=(candidate>=goal)&(median>.1);low=np.where(ok,mid,low);high=np.where(ok,high,mid)
 result['underopen-50'][:,jaw]-=low.astype(np.float32)
 median=np.median(gaps,-1);closed=median<=.1
 # Smoothly hold selected contacts open, including a two-frame 80ms shoulder.
 envelope=closure_mask.astype(float)
 for lag,amount in [(1,.75),(2,.25)]:
  envelope[lag:]=np.maximum(envelope[lag:],amount*closure_mask[:-lag]);envelope[:-lag]=np.maximum(envelope[:-lag],amount*closure_mask[lag:])
 result['missing-closure'][:,jaw]=np.minimum(1,weights[:,jaw]+.18*envelope)
 return result,closure_mask

def main():
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=rig['gap_basis'];rest=rig['rest']+rig['gap_offset'];names=json.loads((WORK/'arkit-student/channels.json').read_text());jaw=names.index('jawOpen');rows=[r for r in json.loads((CORPUS/'manifest.json').read_text()) if r['condition']=='original'];voices=sorted({r['engine']+':'+r['voice'] for r in rows});seen=set();cases=[]
 for split in ['calibration','test-a','test-b']:
  for voice in voices:
   choices=sorted([r for r in rows if r['engine']+':'+r['voice']==voice and r['text'] not in seen],key=lambda r:hashlib.sha256(('expert-probe-73:'+split+r['text']).encode()).hexdigest())
   for row in choices:
    pcm=np.fromfile(row['audio'],'<f4');n=len(pcm)//640
    if n<60:continue
    with np.load(CORPUS/'labels'/(row['stem']+'.npz')) as z:raw=z['weights'];rawg=z['gaps']
    t=np.arange(n)*60/25;left=np.minimum(t.astype(int),len(raw)-1);right=np.minimum(left+1,len(raw)-1);w=raw[left]+(t-left)[:,None]*(raw[right]-raw[left]);w=w.astype(np.float32)
    # Keep head/gaze/eyebrows constant so comparisons isolate mouth/jaw animation.
    w[:,[i for i,name in enumerate(names) if not name.startswith(('mouth','jaw'))]]=0
    gaps=np.einsum('tc,ck->tk',w,basis,optimize=False)+rest;closed=np.median(gaps,-1)<=.1;opened=np.median(gaps,-1)>.35
    if closed[10:-10].sum()<3 or opened.sum()<10:continue
    associations=associate(row['phones'],runs(np.median(rawg,-1)<=.1),len(rawg));events=[v for v in associations if v['status']=='unique'];closure_mask=np.zeros(n,bool)
    for event in events:
     a,b=event['event'];closure_mask |= (np.arange(n)/25>=a/60)&(np.arange(n)/25<b/60)&closed
    if closure_mask.sum()<2:continue
    vv,mask=variants(w,gaps,jaw,basis[jaw],closure_mask);under=np.einsum('tc,ck->tk',vv['underopen-50'],basis,optimize=False)+rest;missing=np.einsum('tc,ck->tk',vv['missing-closure'],basis,optimize=False)+rest
    assert np.isfinite(gaps).all() and np.isfinite(under).all() and np.isfinite(missing).all();np.testing.assert_array_equal(vv['underopen-50'][closed],w[closed]);reduced=(np.median(gaps,-1)-np.median(under,-1))[opened];lifted=np.median(missing,-1)[mask]
    if np.median(reduced)<.15 or np.min(lifted)<.3:continue
    break
   else:raise ValueError('No suitable distinct clip '+voice+' '+split)
   case=dict(id=f'case-{len(cases):02}',split='calibration' if split=='calibration' else 'test',engine=row['engine'],voice=row['voice'],text=row['text'],source=row['stem'],audio=row['audio'],audio_sha256=sha(row['audio']),label_sha256=sha(CORPUS/'labels'/(row['stem']+'.npz')),frames=n,closed_frames=int(closed.sum()),targeted_closure_frames=int(mask.sum()),closure_events=events,underopen_median_reduction_mm=float(10*np.median(reduced)),missing_min_gap_mm=float(10*np.min(lifted)))
   for kind,a in vv.items():freeze(OUT/'tracks'/(case['id']+'-'+kind+'.f32'),a.astype('<f4').tobytes())
   freeze(OUT/'audio'/(case['id']+'.f32'),pcm[:n*640].astype('<f4').tobytes());cases.append(case);seen.add(row['text'])
 freeze(OUT/'cases.json',cases);print('Prepared 4 calibration +8 held-out test clips;',sum(r['frames'] for r in cases),'frames per view.',flush=True)
 print([(r['id'],r['engine'],r['voice'],r['frames'],round(r['underopen_median_reduction_mm'],2)) for r in cases],flush=True)
if __name__=='__main__':main()

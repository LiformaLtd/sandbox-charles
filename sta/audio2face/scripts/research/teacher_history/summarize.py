"""Report identical-input history/noise sensitivity, including temporal decay and controls."""
from itertools import combinations
import numpy as np
from common import *
from geometry import prepare

def metrics(delta_surface,delta_gaps,a_contact=None,b_contact=None):
 result=dict(surface_rms_mm=float(10*np.sqrt(np.square(delta_surface.astype(np.float64)).sum(-1).mean())),gap_rms_mm=float(10*np.sqrt(np.square(delta_gaps.astype(np.float64)).mean())))
 if a_contact is not None:result['contact_disagreement']=float(np.mean(a_contact!=b_contact))
 return result
def aggregate(rows):
 result=dict(n=len(rows))
 for metric in ['surface_rms_mm','gap_rms_mm']:
  values=np.array([r[metric] for r in rows]);result[metric]=float(np.sqrt(np.mean(values**2)));result[metric+'_median']=float(np.median(values));result[metric+'_p95']=float(np.quantile(values,.95));result[metric+'_max']=float(values.max())
 if all('contact_disagreement' in r for r in rows):result['contact_disagreement']=float(np.mean([r['contact_disagreement'] for r in rows]))
 return result

def main():
 protocol=json.loads((OUT/'protocol.json').read_text());cases=json.loads((OUT/'cases.json').read_text());bins=[tuple(b) for b in protocol['bins']]+[tuple(protocol['primary'])];time=np.arange(225)/60;rows=[];floors=[];student_errors=[];curves=[];rig=prepare()
 for case in cases:
  tracks={}
  for history in protocol['histories']:
   for seed in protocol['seeds']:
    with np.load(OUT/'tracks'/case['id']/(history+'-s'+str(seed)+'.npz')) as z:tracks[history,seed]={k:z[k] for k in ['surface','gaps']}
  for seed in protocol['seeds']:
   for a,b in combinations(protocol['histories'],2):
    x,y=tracks[a,seed],tracks[b,seed]
    for lo,hi in bins:
     take=(time>=lo)&(time<hi);rows.append(dict(case=case['id'],engine=case['engine'],voice=case['voice'],kind='history',seed=seed,a=a,b=b,bin=[lo,hi],**metrics(x['surface'][take]-y['surface'][take],x['gaps'][take]-y['gaps'][take],np.median(x['gaps'][take],-1)<=.1,np.median(y['gaps'][take],-1)<=.1)))
    if seed==0:curves.append(dict(case=case['id'],kind='history',a=a,b=b,surface_mm=(10*np.sqrt(np.square(x['surface']-y['surface']).sum(-1).mean(-1))).tolist(),gap_mm=(10*np.sqrt(np.square(x['gaps']-y['gaps']).mean(-1))).tolist()))
  for history in protocol['histories']:
   for seed in [1,2,3]:
    x,y=tracks[history,0],tracks[history,seed]
    for lo,hi in bins:
     take=(time>=lo)&(time<hi);rows.append(dict(case=case['id'],engine=case['engine'],voice=case['voice'],kind='noise',seed=seed,a=history,b=history,bin=[lo,hi],**metrics(x['surface'][take]-y['surface'][take],x['gaps'][take]-y['gaps'][take],np.median(x['gaps'][take],-1)<=.1,np.median(y['gaps'][take],-1)<=.1)))
    if history=='cold':curves.append(dict(case=case['id'],kind='noise',a='0',b=str(seed),surface_mm=(10*np.sqrt(np.square(x['surface']-y['surface']).sum(-1).mean(-1))).tolist(),gap_mm=(10*np.sqrt(np.square(x['gaps']-y['gaps']).mean(-1))).tolist()))
  for history in ['speech-a','speech-b']:
   for seed in [1,2,3]:
    d=tracks[history,seed]['surface']-tracks['silence',seed]['surface']-tracks[history,0]['surface']+tracks['silence',0]['surface'];g=tracks[history,seed]['gaps']-tracks['silence',seed]['gaps']-tracks[history,0]['gaps']+tracks['silence',0]['gaps']
    for lo,hi in bins:
     take=(time>=lo)&(time<hi);rows.append(dict(case=case['id'],engine=case['engine'],voice=case['voice'],kind='interaction',seed=seed,a=history,b='silence',bin=[lo,hi],**metrics(d[take],g[take])))
  for names,label in [(protocol['histories'],'all-history'),(['silence','speech-a','speech-b'],'warm-history')]:
   surface=np.stack([tracks[h,0]['surface'] for h in names]).astype(np.float64);gaps=np.stack([tracks[h,0]['gaps'] for h in names]).astype(np.float64)
   for lo,hi in bins:
    take=(time>=lo)&(time<hi);floors.append(dict(case=case['id'],kind=label,bin=[lo,hi],**metrics(surface[:,take]-surface.mean(0)[None,take],gaps[:,take]-gaps.mean(0)[None,take])))
  pred=np.load(OUT/'tracks'/case['id']/'student.npy')[:,rig['active']];actual=tracks['cold',0];dense=np.einsum('tc,cva->tva',pred,rig['basis'],optimize=False)-rig['offset']
  with np.load(WORK/'arkit-student/rig-metrics.npz') as z:gap_basis=z['gap_basis'][rig['active']];rest=z['rest']+z['gap_offset']
  gaps=np.einsum('tc,ck->tk',pred,gap_basis,optimize=False)+rest
  for lo,hi in bins:
   take=(time>=lo)&(time<hi);student_errors.append(dict(case=case['id'],bin=[lo,hi],**metrics(dense[take]-actual['surface'][take],gaps[take]-actual['gaps'][take])))
 groups={}
 for lo,hi in bins:
  label=f'{lo:g}-{hi:g}'
  chosen=lambda record:record['bin']==[lo,hi]
  group={}
  for a,b in combinations(protocol['histories'],2):group['history-s0:'+a+':'+b]=aggregate([r for r in rows if chosen(r) and r['kind']=='history' and r['seed']==0 and r['a']==a and r['b']==b])
  group['history-s0:warm-pairs']=aggregate([r for r in rows if chosen(r) and r['kind']=='history' and r['seed']==0 and r['a']!='cold' and r['b']!='cold'])
  for history in protocol['histories']:group['noise:'+history]=aggregate([r for r in rows if chosen(r) and r['kind']=='noise' and r['a']==history])
  group['interaction']=aggregate([r for r in rows if chosen(r) and r['kind']=='interaction'])
  for kind in ['all-history','warm-history']:group['conditional-floor:'+kind]=aggregate([r for r in floors if chosen(r) and r['kind']==kind])
  group['student-vs-cold-s0']=aggregate([r for r in student_errors if chosen(r)])
  groups[label]=group
 result=dict(groups=groups,comparisons=rows,conditional_floors=floors,student_errors=student_errors,curves=curves,source_sha256=sha(__file__),interpretation='Constructed matched-window diagnostic; pooled squared-error RMS, not production-distribution irreducible bounds; seeds are fixed per track.')
 freeze(OUT/'summary.json',result);print(json.dumps(groups,indent=2),flush=True)
if __name__=='__main__':main()

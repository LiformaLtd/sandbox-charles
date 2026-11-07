from pathlib import Path
import os
import json
import numpy as np
APP=Path(__file__).resolve().parents[3]
H=Path(os.environ.get('A2F_ABLATION_OUTPUT',APP/'.local/compact-followup')).resolve()
SEEDS=[37,73,101];WIDTHS=[96,128,192]
GROUPS=['new','new_subtitles','new_fresh','prior_v2','prior_v2_natural','original']
METRICS=['lip_gap_rmse','contact_f1','contact_precision','contact_recall','lip_gap_velocity_mae','coefficient_mae','mouth_mae']
selection=lambda scores: scores[0][0]+.6*scores[1][0]+.25*scores[2][0]-.3*scores[0][1]-.2*scores[1][1]-.1*scores[2][1]
training={};quality={};fresh={};result={'seeds':SEEDS,'epochs':12,'widths':{},'paired':{},'selected_by_validation':{}}
for w in WIDTHS:
 names=[f'w{w}-s{s}' for s in SEEDS]
 for n in names:
  training[n]=json.loads((H/(n+'.json')).read_text());quality[n]=json.loads((H/(n+'-quality.json')).read_text());fresh[n]=json.loads((H/('confirmation-'+n+'.json')).read_text())
  assert 'bytes' in training[n]
 scores={n:selection(training[n]['best_dev']) for n in names}
 selected=max(names,key=lambda n:scores[n]);result['selected_by_validation'][str(w)]=selected
 row={'parameters':training[names[0]]['parameters'],'bytes':quality[names[0]]['bytes'],'selected_name':selected,'runs':{}}
 for n in names:
  row['runs'][n]={'best_epoch':training[n]['best_epoch'],'validation_score':scores[n],'sha256':quality[n]['sha256'],'history':training[n]['history'],'confirmation_phone_events':fresh[n]['phone_events']}
 for g in GROUPS+['confirmation']:
  values=[(fresh[n]['all'] if g=='confirmation' else quality[n][g])['student'] for n in names]
  row[g]={m:{'mean':float(np.mean([v[m] for v in values])),'sd':float(np.std([v[m] for v in values],ddof=1)),'values':[v[m] for v in values]} for m in METRICS}
 result['widths'][str(w)]=row

# Crossed paired bootstrap: resample matched seeds and shared test clips.
# Keep complete clips and use the same draws for both widths.
def bootstrap(reference,candidate,group):
 fields=['squared_error','values','tp','actual','closed'];arrays=[]
 for width in (reference,candidate):
  seed_rows=[]
  for seed in SEEDS:
   name=f'w{width}-s{seed}'
   rows=fresh[name]['per_clip'] if group=='confirmation' else quality[name]['per_clip'][group]
   rows=sorted(rows,key=lambda r:r['stem'])
   seed_rows.append([[r[f] for f in fields] for r in rows])
  arrays.append(np.asarray(seed_rows))
 a,b=arrays
 rng=np.random.default_rng(37);draws=4000
 seed_ids=rng.integers(0,3,(draws,3));clips=rng.integers(0,a.shape[1],(draws,a.shape[1]))
 aa=a[seed_ids[:,:,None],clips[:,None,:]].sum(2);bb=b[seed_ids[:,:,None],clips[:,None,:]].sum(2)
 rmse=(np.sqrt(bb[:,:,0]/bb[:,:,1])-np.sqrt(aa[:,:,0]/aa[:,:,1])).mean(1)
 f1=(2*bb[:,:,2]/np.maximum(1,bb[:,:,3]+bb[:,:,4])-2*aa[:,:,2]/np.maximum(1,aa[:,:,3]+aa[:,:,4])).mean(1)
 def points(x):return {'delta_95ci':np.quantile(x,[.025,.975]).tolist()}
 return {'rmse':points(rmse),'f1':points(f1),'draws':draws,'units':'paired seeds and whole clips'}
for w in (128,192):
 result['paired'][str(w)]={g:bootstrap(96,w,g) for g in ['new','prior_v2','original','confirmation']}
 (result['paired'][str(w)])['per_seed_confirmation_delta']={str(seed):{m:fresh[f'w{w}-s{seed}']['all']['student'][m]-fresh[f'w96-s{seed}']['all']['student'][m] for m in ['lip_gap_rmse','contact_f1']} for seed in SEEDS}
result['baseline_confirmation']=json.loads((H/'confirmation-baseline.json').read_text())
(H/'study-summary.json').write_text(json.dumps(result,indent=2))
for width,row in result['widths'].items():
 print(width,'selected',row['selected_name'],'new',row['new']['lip_gap_rmse'],row['new']['contact_f1'],'confirmation',row['confirmation']['lip_gap_rmse'],row['confirmation']['contact_f1'])

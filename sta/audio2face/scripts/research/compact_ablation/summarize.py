from pathlib import Path
import os
import json
import numpy as np
APP=Path(__file__).resolve().parents[3]
H=Path(os.environ.get('A2F_ABLATION_OUTPUT',APP/'.local/compact-ablation')).resolve()
names=['baseline','control','width128','width192','context1500','contextwide','data']
quality={n:json.loads((H/f'{n}-quality.json').read_text()) for n in names}

def paired_ci(reference,candidate,group):
    rows={r['stem']:r for r in reference['per_clip'][group]}
    other={r['stem']:r for r in candidate['per_clip'][group]}
    assert rows.keys()==other.keys()
    fields=['squared_error','values','tp','actual','closed']
    a,b=[np.array([[d[k][f] for f in fields] for k in rows]) for d in (rows,other)]
    ids=np.random.default_rng(37).integers(0,len(a),size=(2000,len(a)))
    a,b=a[ids].sum(1),b[ids].sum(1)
    delta_rmse=np.sqrt(b[:,0]/b[:,1])-np.sqrt(a[:,0]/a[:,1])
    delta_f1=2*b[:,2]/np.maximum(1,b[:,3]+b[:,4])-2*a[:,2]/np.maximum(1,a[:,3]+a[:,4])
    return {'rmse_delta_95ci':np.quantile(delta_rmse,[.025,.975]).tolist(),'f1_delta_95ci':np.quantile(delta_f1,[.025,.975]).tolist()}
result={}
for name in names:
    q=quality[name]
    result[name]={'bytes':q['bytes'],'sha256':q['sha256'],'phone_events':q['phone_events']}
    for group in ['new','new_subtitles','new_fresh','prior_v2','prior_v2_natural','original']:
        result[name][group]=q[group]['student']
        if name not in ('baseline',):
            result[name][group]['versus_baseline_ci']=paired_ci(quality['baseline'],q,group)
        if name not in ('baseline','control'):
            result[name][group]['versus_control_ci']=paired_ci(quality['control'],q,group)
    if name!='baseline':
        config=json.loads((H/(name+'.json')).read_text())
        result[name]['training']={k:config[k] for k in ['config','best_epoch','parameters','seconds','optimizer_steps_per_epoch','new_windows','old_windows']}
    runs=[json.loads(p.read_text())['summary'] for p in sorted(H.glob('browser-'+name+'-*.json'))]
    if runs:
        result[name]['browser']={}
        for length in ('short','long'):
            subset=[r for r in runs if r['length']==length]
            if not subset:continue
            result[name]['browser'][length]={'repeats':len(subset),**{key:float(np.median([r[key] for r in subset])) for key in ['peak_MB','block_ms','first_block_ms','time_to_audio_ms','gpu_peak_bytes']},'idle_after':all(r['idle_after'] for r in subset)}
(H/'summary.json').write_text(json.dumps(result,indent=2))
for name,row in result.items():
    print(name,round(row['bytes']/1e6,3),'MB',[(g,row[g]['contact_f1'],row[g]['lip_gap_rmse']) for g in ['new','prior_v2','original']],row.get('browser',{}),flush=True)

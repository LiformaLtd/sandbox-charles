"""Report every fixed seed; crossed paired bootstrap is exploratory, not new data."""
import json
import numpy as np
from prepare_corpus import OUT,freeze,sha


def main():
    roots=[OUT,OUT/'repeats/s73',OUT/'repeats/s101'];seeds=[37,73,101];rows=[];reports=[]
    for seed,root in zip(seeds,roots):
        report={name:json.loads((root/('evaluation-'+name+'.json')).read_text()) for name in ['v4','original-long','expanded-long']};reports.append(report)
        c,b,v=[report[n]['fresh']['conditions']['original'] for n in ['expanded-long','original-long','v4']]
        rows.append(dict(seed=seed,control_rms_mm=b['macro_gap_rmse_mm'],candidate_rms_mm=c['macro_gap_rmse_mm'],v4_rms_mm=v['macro_gap_rmse_mm'],
            improvement_vs_control=1-c['macro_gap_rmse_mm']/b['macro_gap_rmse_mm'],improvement_vs_v4=1-c['macro_gap_rmse_mm']/v['macro_gap_rmse_mm'],control_f1=b['macro_contact_f1'],candidate_f1=c['macro_contact_f1'],
            engine_improvements={e:1-c['engines'][e]['macro_gap_rmse_mm']/b['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon','supertonic']},
            control_well_timed={p:b['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']},candidate_well_timed={p:c['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']}))
    primary=lambda name:float(np.mean([r[name] for r in rows]))
    mean=dict(control_rms_mm=primary('control_rms_mm'),candidate_rms_mm=primary('candidate_rms_mm'),v4_rms_mm=primary('v4_rms_mm'),
        improvement_vs_control=1-primary('candidate_rms_mm')/primary('control_rms_mm'),improvement_vs_v4=1-primary('candidate_rms_mm')/primary('v4_rms_mm'),control_f1=primary('control_f1'),candidate_f1=primary('candidate_f1'),
        control_well_timed={p:float(np.mean([r['control_well_timed'][p] for r in rows])) for p in ['m','b','p']},candidate_well_timed={p:float(np.mean([r['candidate_well_timed'][p] for r in rows])) for p in ['m','b','p']})
    mean['v4_f1']=reports[0]['v4']['fresh']['conditions']['original']['macro_contact_f1']
    mean['v4_well_timed']={p:reports[0]['v4']['fresh']['conditions']['original']['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']}
    mean['engines']={}
    for engine in ['phonon','supertonic']:
        scores={name:float(np.mean([r[name]['fresh']['conditions']['original']['engines'][engine]['macro_gap_rmse_mm'] for r in reports])) for name in ['v4','original-long','expanded-long']}
        mean['engines'][engine]=dict(rms_mm=scores,improvement_vs_control=1-scores['expanded-long']/scores['original-long'],improvement_vs_v4=1-scores['expanded-long']/scores['v4'])
    # Full crossing: three seeds, four presets, 88 original transcript recordings.
    voices=sorted(reports[0]['v4']['fresh']['conditions']['original']['voices'])
    texts=sorted({r['text'] for r in reports[0]['v4']['fresh']['clips'] if r['condition']=='original'})
    arrays={}
    for model in ['original-long','expanded-long']:
        a=np.zeros((3,4,len(texts),5),np.float64);seen=set()
        for s,report in enumerate(reports):
            for clip in report[model]['fresh']['clips']:
                if clip['condition']!='original':continue
                v=voices.index(clip['engine']+':'+clip['voice']);t=texts.index(clip['text']);seen.add((s,v,t))
                a[s,v,t]=[clip[k] for k in ['squared_gap_error','gap_values','contact_hits','teacher_contacts','student_contacts']]
        if len(seen)!=3*4*len(texts):raise ValueError('Incomplete crossed confirmation')
        arrays[model]=a
    rng=np.random.default_rng(20261107);draws=[]
    for _ in range(2000):
        seed_ix=rng.integers(0,3,3);text_ix=rng.integers(0,len(texts),len(texts))
        voice_ix=np.r_[rng.integers(0,2,2),rng.integers(2,4,2)]
        values=[]
        for model in arrays:
            sampled=arrays[model][seed_ix][:,voice_ix][:,:,text_ix].sum(axis=2)
            rms=float(np.mean(10*np.sqrt(sampled[...,0]/sampled[...,1])))
            f1=float(np.mean(2*sampled[...,2]/np.maximum(1,sampled[...,3]+sampled[...,4])))
            values.append((rms,f1))
        draws.append((1-values[1][0]/values[0][0],values[1][1]-values[0][1]))
    interval=np.quantile(np.array(draws),[.025,.975],axis=0)
    result=dict(post_pilot=True,seeds=rows,mean=mean,exploratory_bootstrap=dict(draws=2000,seed=20261107,
        rms_relative_improvement_95_interval=interval[:,0].tolist(),contact_f1_difference_95_interval=interval[:,1].tolist(),
        method='Paired crossed bootstrap of optimizer seeds, transcripts and voice presets within engine; same inspected cohort, only four presets'),
        original_pilot_screen_unchanged=json.loads((OUT/'summary.json').read_text())['promising'],
        source_sha256=sha(__file__))
    (OUT/'seed-stability-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

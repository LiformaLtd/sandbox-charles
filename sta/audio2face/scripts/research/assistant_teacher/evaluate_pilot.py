"""Locked selections; full primary training fit and inspected regression cohorts."""
from common import *
import numpy as np,onnx,onnxruntime as ort,mlx.core as mx
from onnx import numpy_helper
from model import Assistant,Student,load_selected
from features import Dataset
from evaluate import evaluate_cohort
sys.path.insert(0,str(APP/'scripts/research/closure_protected'))
from inference import infer
import metrics
metrics.infer=infer

def training_fit(net,basis,rest,active):
    stats={};clips=0
    for part in ['old-train','new-train']:
        dataset=Dataset([part]);data=dataset.parts[0];rows=[r for r in json.loads((SCALE/'datasets'/part/'manifest.json').read_text()) if r['split']=='train'];cursor=0
        prediction=np.empty((dataset.length,30,39),np.float32)
        for start in range(0,dataset.length,128):prediction[start:start+128]=np.array(mx.clip(net(mx.array(data['mel'][start:start+128]))[:,15:45],0,1))
        for row in rows:
            n=len(range(-8000,Path(row['audio']).stat().st_size//4-4000,8000));valid=data['valid'][cursor:cursor+n];w=prediction[cursor:cursor+n][valid];truth=data['gaps'][cursor:cursor+n][valid];gap=w@basis[active]+rest;cursor+=n
            a=np.median(truth,-1)<=.1;b=np.median(gap,-1)<=.1;key=(row['condition'],row['engine']+':'+row['voice']);s=stats.setdefault(key,np.zeros(5));s += [np.square(gap-truth).sum(),truth.size,(a&b).sum(),a.sum(),b.sum()];clips+=1
        assert cursor==dataset.length
    result={}
    for condition in sorted({k[0] for k in stats}):
        voices={v:dict(rms_mm=float(10*np.sqrt(s[0]/s[1])),f1=float(2*s[2]/max(1,s[3]+s[4]))) for (c,v),s in stats.items() if c==condition}
        result[condition]=dict(rms_mm=float(np.mean([v['rms_mm'] for v in voices.values()])),f1=float(np.mean([v['f1'] for v in voices.values()])),voices=voices)
    return dict(clips=clips,conditions=result)

def main():
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=rig['gap_basis'];rest=rig['rest']+rig['gap_offset'];active=rig['active'];basevalues={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer}
    models={'v4':BASELINE,'assistant':OUT/'assistant/model.onnx'}
    for arm in ['control','distilled']:
        if (OUT/arm/'selection.json').exists():models[arm]=OUT/arm/'model.onnx'
    selections={a:json.loads((OUT/a/'selection.json').read_text()) for a in models if a!='v4'}
    for a,p in models.items():
        if a!='v4':assert sha(p)==selections[a]['sha256']
    freeze(OUT/'evaluation-lock.json',dict(source_sha256=sha(__file__),models={a:sha(p) for a,p in models.items()},selections=selections,scope='Inspected regression cohorts, not new independent confirmation. Training scores never select checkpoints.'))
    reports={}
    for arm,path in models.items():
        dest=OUT/('evaluation-'+arm+'.json')
        if dest.exists():reports[arm]=json.loads(dest.read_text());continue
        if arm=='assistant':net=Assistant(basevalues);load_selected(net,OUT/'assistant/best.npz')
        else:net=Student({v.name:numpy_helper.to_array(v).copy() for v in onnx.load(path).graph.initializer})
        report=dict(sha256=sha(path),bytes=path.stat().st_size,training=training_fit(net,basis,rest,active))
        option=ort.SessionOptions();option.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=option,providers=['CPUExecutionProvider']);cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True)
        for name,root in [('protected_regression',APP/'.local/compact-protected-v1/confirmation/variants'),('feature_regression',APP/'.local/compact-feature-v1/confirmation/variants'),('historical',APP/'.local/compact-gain-v2')]:
            rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='test'];report[name]=evaluate_cohort(session,rows,root/'labels',cache,basis,rest,active)
            print(arm,name,report[name]['conditions']['original']['macro_gap_rmse_mm'],flush=True)
        freeze(dest,report);reports[arm]=report
    condition=lambda a:reports[a]['protected_regression']['conditions']['original']
    summary={a:dict(training_original=r['training']['conditions']['original'],regression_original=condition(a)) for a,r in reports.items()}
    if 'distilled' in reports:
        control,treat,release=[condition(a) for a in ['control','distilled','v4']];hits=lambda r,p:r['closure_audit'][p]['well_timed_hits']
        checks=dict(extra_gap_gain=1-treat['macro_gap_rmse_mm']/control['macro_gap_rmse_mm']>=.03,contact_f1=treat['macro_contact_f1']>=max(control['macro_contact_f1'],release['macro_contact_f1'])-1e-6,phone_hits=all(hits(treat,p)>=max(hits(control,p),hits(release,p)) for p in ['m','b','p']),engine_f1=all(treat['engines'][e]['macro_contact_f1']>=max(control['engines'][e]['macro_contact_f1'],release['engines'][e]['macro_contact_f1'])-1e-6 for e in ['phonon','supertonic']))
        def rates(arm):
            voices={}
            for row in reports[arm]['protected_regression']['clips']:
                if row['condition']!='original':continue
                s=voices.setdefault(row['engine']+':'+row['voice'],np.zeros(4));s += [row['student_contacts']-row['contact_hits'],row['gap_values']/9-row['teacher_contacts'],row['teacher_contacts']-row['contact_hits'],row['teacher_contacts']]
            return dict(false_closure=float(np.mean([s[0]/max(1,s[1]) for s in voices.values()])),missed_contact=float(np.mean([s[2]/max(1,s[3]) for s in voices.values()])))
        for key in ['false_closure','missed_contact']:checks[key]=rates('distilled')[key]<=rates('control')[key]+1e-6
        for key in ['onset_errors_ms','release_errors_ms']:
            timing=lambda r:float(np.mean([r['closure_audit'][p][key]['mae'] for p in ['m','b','p']]))
            checks[key]=timing(treat)<=timing(control)+.1
        for cohort in ['feature_regression','historical']:
            t,c=[reports[a][cohort]['conditions']['original'] for a in ['distilled','v4']];checks[cohort]=t['macro_gap_rmse_mm']<=1.02*c['macro_gap_rmse_mm'] and t['macro_contact_f1']>=c['macro_contact_f1']-.01
        freeze(OUT/'screen.json',dict(checks={k:bool(v) for k,v in checks.items()},promising=all(checks.values()),extra_gap_improvement=1-treat['macro_gap_rmse_mm']/control['macro_gap_rmse_mm'],rates={a:rates(a) for a in reports},note='Exploratory inspected-regression screen, not release confirmation.'))
    freeze(OUT/'quality-summary.json',summary)
    print(json.dumps({a:dict(train=r['training_original']['rms_mm'],test=r['regression_original']['macro_gap_rmse_mm'],f1=r['regression_original']['macro_contact_f1']) for a,r in summary.items()},indent=2))

if __name__=='__main__':main()

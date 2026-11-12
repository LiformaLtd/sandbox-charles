"""Locked fresh-speech comparison, training fit and inspected regressions."""
from common import *
import numpy as np,onnx,onnxruntime as ort,mlx.core as mx
from onnx import numpy_helper
from residual_model import Head
from cache_features import FrozenDataset as Dataset
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

def rates(report):
    voices={}
    for row in report['fresh']['clips']:
        if row['condition']!='original':continue
        s=voices.setdefault(row['engine']+':'+row['voice'],np.zeros(4));s += [row['student_contacts']-row['contact_hits'],row['gap_values']/9-row['teacher_contacts'],row['teacher_contacts']-row['contact_hits'],row['teacher_contacts']]
    return dict(false_closure=float(np.mean([s[0]/max(1,s[1]) for s in voices.values()])),missed_contact=float(np.mean([s[2]/max(1,s[3]) for s in voices.values()])))

def main():
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=rig['gap_basis'];rest=rig['rest']+rig['gap_offset'];active=rig['active'];values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};scale=np.load(OUT/'feature-scale.npy')
    models={'v4':BASELINE,**{a:OUT/a/'model.onnx' for a in ['control','residual']}};selections={a:json.loads((OUT/a/'selection.json').read_text()) for a in ['control','residual']}
    for a,r in selections.items():assert sha(models[a])==r['sha256']
    assert selections['control']['sampler_sha256']==selections['residual']['sampler_sha256'];freeze(OUT/'selection-lock.json',selections)
    complete=json.loads((OUT/'confirmation/complete.json').read_text());assert complete['manifest_sha256']==sha(OUT/'confirmation/variants/manifest.json');assert complete['recipe_sha256']==sha(OUT/'confirmation/variants/label-recipe.json')
    freeze(OUT/'evaluation-lock.json',dict(source_sha256=sha(__file__),models={a:sha(p) for a,p in models.items()},confirmation_sha256=sha(OUT/'confirmation/complete.json'),scorer_sha256=sha(APP/'scripts/research/multivoice/closure_metrics.py')))
    reports={}
    for arm,path in models.items():
        dest=OUT/('evaluation-'+arm+'.json')
        if dest.exists():reports[arm]=json.loads(dest.read_text());continue
        net=Head(values,scale,arm if arm!='v4' else 'residual')
        if arm!='v4':net.restore(OUT/arm/'best.npz')
        report=dict(sha256=sha(path),bytes=path.stat().st_size,training=training_fit(net,basis,rest,active));option=ort.SessionOptions();option.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=option,providers=['CPUExecutionProvider']);cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True)
        for name,root in [('fresh',OUT/'confirmation/variants'),('feature_regression',FEATURE/'confirmation/variants'),('historical',APP/'.local/compact-gain-v2')]:
            rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='test'];report[name]=evaluate_cohort(session,rows,root/'labels',cache,basis,rest,active);print(arm,name,report[name]['conditions']['original']['macro_gap_rmse_mm'],flush=True)
        freeze(dest,report);reports[arm]=report
    original=lambda a:reports[a]['fresh']['conditions']['original']
    c,r,v=[original(a) for a in ['control','residual','v4']];hits=lambda x,p:x['closure_audit'][p]['well_timed_hits']
    checks=dict(extra_gap_gain=1-r['macro_gap_rmse_mm']/c['macro_gap_rmse_mm']>=.03,release_gap=r['macro_gap_rmse_mm']<v['macro_gap_rmse_mm'],engine_gap=all(r['engines'][e]['macro_gap_rmse_mm']<=c['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon','supertonic']),f1=r['macro_contact_f1']>=max(c['macro_contact_f1'],v['macro_contact_f1'])-1e-6,engine_f1=all(r['engines'][e]['macro_contact_f1']>=max(c['engines'][e]['macro_contact_f1'],v['engines'][e]['macro_contact_f1'])-1e-6 for e in ['phonon','supertonic']),phone_hits=all(hits(r,p)>=max(hits(c,p),hits(v,p)) for p in ['m','b','p']))
    error_rates={a:rates(t) for a,t in reports.items()};timing={a:{k:float(np.mean([original(a)['closure_audit'][p][k]['mae'] for p in ['m','b','p']])) for k in ['onset_errors_ms','release_errors_ms']} for a in reports}
    for key in ['false_closure','missed_contact']:checks[key]=error_rates['residual'][key]<=error_rates['control'][key]+1e-6
    for key in ['onset_errors_ms','release_errors_ms']:checks[key]=timing['residual'][key]<=timing['control'][key]+.1
    for cohort in ['feature_regression','historical']:
        a,b=[reports[n][cohort]['conditions']['original'] for n in ['residual','v4']];checks[cohort]=a['macro_gap_rmse_mm']<=1.02*b['macro_gap_rmse_mm'] and a['macro_contact_f1']>=b['macro_contact_f1']-.01
    freeze(OUT/'screen.json',dict(checks={k:bool(v) for k,v in checks.items()},promising=all(checks.values()),extra_gap_improvement=1-r['macro_gap_rmse_mm']/c['macro_gap_rmse_mm'],error_rates=error_rates,timing=timing,hits={a:{p:hits(original(a),p) for p in ['m','b','p']} for a in reports}))
    freeze(OUT/'quality-summary.json',{a:dict(training=r['training']['conditions'],fresh=r['fresh']['conditions']) for a,r in reports.items()});print((OUT/'screen.json').read_text(),flush=True)

if __name__=='__main__':main()

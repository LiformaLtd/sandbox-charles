"""Locked checkpoint evaluation; fresh confirmation is primary, prior sets regressions."""
from common import *
import numpy as np,onnxruntime as ort
from inference import infer
from evaluate import evaluate_cohort
import metrics
metrics.infer=infer

def error_rates(report):
 groups={}
 for row in report['fresh']['clips']:
  if row['condition']!='original':continue
  g=groups.setdefault(row['engine']+':'+row['voice'],np.zeros(4));g += [row['student_contacts']-row['contact_hits'],row['gap_values']/9-row['teacher_contacts'],row['teacher_contacts']-row['contact_hits'],row['teacher_contacts']]
 return dict(false_closure_rate=float(np.mean([g[0]/max(g[1],1) for g in groups.values()])),missed_contact_rate=float(np.mean([g[2]/max(g[3],1) for g in groups.values()])))

def timing_errors(condition):
 return {kind:float(np.mean([condition['closure_audit'][p][kind]['mae'] for p in ['m','b','p']])) for kind in ['onset_errors_ms','release_errors_ms']}

def main():
 selections={a:json.loads((OUT/a/'selection.json').read_text()) for a in ['control','protected']}
 for a,r in selections.items():assert sha(OUT/a/'model.onnx')==r['model_sha256']
 assert json.loads((OUT/'trajectories/control/complete.json').read_text())['sampler_sha256']==json.loads((OUT/'trajectories/protected/complete.json').read_text())['sampler_sha256']
 freeze(OUT/'selection-lock.json',selections);freeze(OUT/'evaluation-lock.json',dict(sources={str(p.relative_to(APP)):sha(p) for p in [HERE/'evaluate_experiment.py',HERE/'inference.py',APP/'scripts/research/data_scale/evaluate.py',APP/'scripts/research/multivoice/closure_metrics.py']},manifest_sha256=sha(OUT/'confirmation/variants/manifest.json')));rig=np.load(WORK/'arkit-student/rig-metrics.npz');reports={}
 for arm,path in [('v4',APP/'.local/compact-feature-v1/control/model.onnx'),('start',BASELINE),('control',OUT/'control/model.onnx'),('protected',OUT/'protected/model.onnx')]:
  opt=ort.SessionOptions();opt.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider']);cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True);report=dict(model_sha256=sha(path),bytes=path.stat().st_size)
  for name,root in [('fresh',OUT/'confirmation/variants'),('surface_regression',APP/'.local/compact-surface-v1/confirmation/variants'),('feature_regression',FEATURE/'confirmation/variants'),('historical',APP/'.local/compact-gain-v2'),('recent_regression',APP/'.local/compact-recurrent-v1/confirmation/variants')]:
   rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='test'];report[name]=evaluate_cohort(session,rows,root/'labels',cache,rig['gap_basis'],rig['rest']+rig['gap_offset'],rig['active'])
   print(arm,name,report[name]['conditions']['original']['macro_gap_rmse_mm'],flush=True)
  reports[arm]=report;freeze(OUT/('evaluation-'+arm+'.json'),report)
 def condition(arm,cohort='fresh'):return reports[arm][cohort]['conditions']['original']
 c,r,v=[condition(a) for a in ['control','protected','v4']];hits={a:{p:condition(a)['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']} for a in reports}
 checks=dict(gap=1-r['macro_gap_rmse_mm']/c['macro_gap_rmse_mm']>=.03,engines=all(r['engines'][e]['macro_gap_rmse_mm']<=c['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon','supertonic']),f1=r['macro_contact_f1']>=c['macro_contact_f1']-1e-6,closures=sum(hits['protected'].values())>=sum(hits['control'].values()),phone=min(hits['protected'][p]-hits['control'][p] for p in ['m','b','p'])>=0)
 rates={a:error_rates(reports[a]) for a in reports};times={a:timing_errors(condition(a)) for a in reports}
 for key in ['false_closure_rate','missed_contact_rate']:checks[key]=rates['protected'][key]<=rates['control'][key]+1e-6
 for key in ['onset_errors_ms','release_errors_ms']:checks[key]=times['protected'][key]<=times['control'][key]+.1
 checks['engine_contacts']=all(r['engines'][e]['macro_contact_f1']>=c['engines'][e]['macro_contact_f1']-1e-6 for e in ['phonon','supertonic'])
 for name in ['historical','feature_regression']:checks[name]=condition('protected',name)['macro_gap_rmse_mm']<=1.02*condition('v4',name)['macro_gap_rmse_mm'] and condition('protected',name)['macro_contact_f1']>=condition('v4',name)['macro_contact_f1']-.01
 freeze(OUT/'summary.json',dict(gap_improvement_vs_control=1-r['macro_gap_rmse_mm']/c['macro_gap_rmse_mm'],gap_improvement_vs_v4=1-r['macro_gap_rmse_mm']/v['macro_gap_rmse_mm'],hits=hits,error_rates=rates,timing_errors=times,checks={k:bool(v) for k,v in checks.items()},promising=bool(all(checks.values()))));print((OUT/'summary.json').read_text(),flush=True)
if __name__=='__main__':main()

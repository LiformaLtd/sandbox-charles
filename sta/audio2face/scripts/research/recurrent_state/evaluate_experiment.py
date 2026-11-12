"""Locked checkpoint evaluation; fresh confirmation is primary, prior sets regressions."""
from common import *
import numpy as np,onnxruntime as ort
from inference import infer
from evaluate import evaluate_cohort
import metrics
metrics.infer=infer

def main():
 selections={a:json.loads((OUT/a/'selection.json').read_text()) for a in ['control','recurrent']}
 for a,r in selections.items():assert sha(OUT/a/'model.onnx')==r['model_sha256']
 assert selections['control']['schedule_sha256']==selections['recurrent']['schedule_sha256']
 freeze(OUT/'selection-lock.json',selections);freeze(OUT/'evaluation-lock.json',dict(sources={str(p.relative_to(APP)):sha(p) for p in [HERE/'evaluate_experiment.py',HERE/'inference.py',APP/'scripts/research/data_scale/evaluate.py',APP/'scripts/research/multivoice/closure_metrics.py']},manifest_sha256=sha(OUT/'confirmation/variants/manifest.json')));rig=np.load(WORK/'arkit-student/rig-metrics.npz');reports={}
 for arm,path in [('v4',BASELINE),('control',OUT/'control/model.onnx'),('recurrent',OUT/'recurrent/model.onnx')]:
  opt=ort.SessionOptions();opt.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider']);cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True);report=dict(model_sha256=sha(path),bytes=path.stat().st_size)
  for name,root in [('fresh',OUT/'confirmation/variants'),('surface_regression',APP/'.local/compact-surface-v1/confirmation/variants'),('feature_regression',FEATURE/'confirmation/variants'),('historical',APP/'.local/compact-gain-v2')]:
   rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='test'];report[name]=evaluate_cohort(session,rows,root/'labels',cache,rig['gap_basis'],rig['rest']+rig['gap_offset'],rig['active'])
   print(arm,name,report[name]['conditions']['original']['macro_gap_rmse_mm'],flush=True)
  reports[arm]=report;freeze(OUT/('evaluation-'+arm+'.json'),report)
 def condition(arm,cohort='fresh'):return reports[arm][cohort]['conditions']['original']
 c,r,v=[condition(a) for a in ['control','recurrent','v4']];hits={a:{p:condition(a)['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']} for a in reports}
 checks=dict(gap=1-r['macro_gap_rmse_mm']/c['macro_gap_rmse_mm']>=.03,engines=all(r['engines'][e]['macro_gap_rmse_mm']<=c['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon','supertonic']),f1=r['macro_contact_f1']>=c['macro_contact_f1']-.002,closures=sum(hits['recurrent'].values())>=sum(hits['control'].values()),phone=min(hits['recurrent'][p]-hits['control'][p] for p in ['m','b','p'])>=-2)
 for name in ['historical','feature_regression']:checks[name]=condition('recurrent',name)['macro_gap_rmse_mm']<=1.02*condition('v4',name)['macro_gap_rmse_mm'] and condition('recurrent',name)['macro_contact_f1']>=condition('v4',name)['macro_contact_f1']-.01
 freeze(OUT/'summary.json',dict(gap_improvement_vs_control=1-r['macro_gap_rmse_mm']/c['macro_gap_rmse_mm'],gap_improvement_vs_v4=1-r['macro_gap_rmse_mm']/v['macro_gap_rmse_mm'],hits=hits,checks={k:bool(v) for k,v in checks.items()},promising=bool(all(checks.values()))));print((OUT/'summary.json').read_text(),flush=True)
if __name__=='__main__':main()

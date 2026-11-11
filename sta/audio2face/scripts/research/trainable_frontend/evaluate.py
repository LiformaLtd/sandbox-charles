"""Lock paired selections, then score existing held-out recordings and gain variants."""
from common import *
import importlib.util,numpy as np,onnxruntime as ort
spec=importlib.util.spec_from_file_location('frontend_scale_evaluation',APP/'scripts/research/data_scale/evaluate.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def main():
 protocol=lock();models={'baseline':BASELINE}
 for seed in protocol['seeds']:
  for arm in protocol['arms']:models[f'{arm}-s{seed}']=OUT/f'{arm}-s{seed}/model.onnx'
 selections={}
 for name,path in models.items():
  if name=='baseline':selections[name]={'sha256':sha(path)};continue
  record=json.loads((path.parent/'complete.json').read_text())
  if sha(path)!=record['export']['sha256']:raise ValueError('Selected model changed')
  selections[name]=record
 for seed in protocol['seeds']:
  a,b=[selections[f'{arm}-s{seed}'] for arm in protocol['arms']]
  for field in ['steps','sampler_sha256','initial_parameters_sha256','protocol_sha256']:
   if a[field]!=b[field]:raise ValueError('Unmatched paired training: '+field)
 write(OUT/'selection-lock.json',selections)
 cohorts={'feature_confirmation':DATA/'.local/compact-feature-v1/confirmation/variants','scale_confirmation':SCALE/'datasets/confirmation','historical':DATA/'.local/compact-gain-v2'}
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');reports={}
 for name,path in models.items():
  dest=OUT/f'evaluation-{name}.json'
  if dest.exists():
   report=json.loads(dest.read_text())
   if report['sha256']!=sha(path):raise ValueError('Cached result changed')
  else:
   options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
   session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True)
   report={'sha256':sha(path)}
   for cohort,root in cohorts.items():
    if sha(root/'manifest.json')!=protocol['heldout_cohorts'][str(root/'manifest.json')]:raise ValueError('Changed held-out manifest')
    rows=[r for r in json.loads((root/'manifest.json').read_text()) if r.get('split')=='test']
    report[cohort]=old.evaluate_cohort(session,rows,root/'labels',cache,rig['gap_basis'],rig['rest']+rig['gap_offset'],rig['active'])
    print(name,cohort,report[cohort]['conditions']['original']['macro_gap_rmse_mm'],flush=True)
   write(dest,report)
  reports[name]=report
 def stats(n,c,condition='original'):return reports[n][c]['conditions'][condition]
 effects={}
 for cohort in cohorts:
  gaps=[];contacts=[];engines=[];opening=[];closures=[];levels=[]
  for seed in protocol['seeds']:
   fixed,learned=[f'{arm}-s{seed}' for arm in protocol['arms']];a,b=stats(fixed,cohort),stats(learned,cohort)
   gaps.append(1-b['macro_gap_rmse_mm']/a['macro_gap_rmse_mm']);contacts.append(b['macro_contact_f1']-a['macro_contact_f1'])
   engines.append({e:1-b['engines'][e]['macro_gap_rmse_mm']/a['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon','supertonic']})
   opening.append(np.mean([v['opening_rmse_mm'] for v in b['voices'].values()])-np.mean([v['opening_rmse_mm'] for v in a['voices'].values()]))
   closures.append(sum(b['closure_audit'][p]['well_timed_hits']-a['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']))
   variants=[g for g in ['half','boost'] if g in reports[fixed][cohort]['conditions']]
   levels.append(float(np.mean([stats(learned,cohort,g)['macro_gap_rmse_mm']-stats(fixed,cohort,g)['macro_gap_rmse_mm'] for g in variants])) if variants else None)
  effects[cohort]=dict(relative_gap_reduction_by_seed=gaps,mean_relative_gap_reduction=float(np.mean(gaps)),contact_f1_delta_by_seed=contacts,mean_contact_f1_delta=float(np.mean(contacts)),engine_effects=engines,opening_rms_delta_mm_by_seed=opening,well_timed_closure_delta_by_seed=closures,gain_condition_rms_delta_mm_by_seed=levels)
 e=effects['feature_confirmation'];checks=dict(gap=e['mean_relative_gap_reduction']>=.03,contact=e['mean_contact_f1_delta']>=0,opening=np.mean(e['opening_rms_delta_mm_by_seed'])<=0,closures=np.mean(e['well_timed_closure_delta_by_seed'])>=0,engines=all(np.mean([x[k] for x in e['engine_effects']])>=0 for k in ['phonon','supertonic']),gain=np.mean(e['gain_condition_rms_delta_mm_by_seed'])<=0)
 for cohort in ['scale_confirmation','historical']:checks[cohort]=np.mean([stats(f'learned-s{s}',cohort)['macro_gap_rmse_mm'] for s in protocol['seeds']])<=1.02*stats('baseline',cohort)['macro_gap_rmse_mm']
 summary=dict(cohort_status=protocol['cohort_status'],effects=effects,checks={k:bool(v) for k,v in checks.items()},promising=bool(all(checks.values())))
 write(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()

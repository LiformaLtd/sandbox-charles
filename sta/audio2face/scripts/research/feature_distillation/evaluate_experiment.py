"""Lock both selections before fresh lip-sync scoring; no test-driven checkpoint changes."""
from common import *
import numpy as np,onnx,onnxruntime as ort
from evaluate import evaluate_cohort

def main():
 locks={};graph=onnx.load(BASELINE);operators=lambda g:[(n.op_type,list(n.input),list(n.output)) for n in g.graph.node]
 for name in ['control','distill']:
  root=OUT/name;row=json.loads((root/'selection.json').read_text());g=onnx.load(root/'model.onnx')
  assert sha(root/'model.onnx')==row['model_sha256'] and row['bytes']==2251827 and row['parameters']==331047
  assert operators(graph)==operators(g) and not any('hint' in v.name for v in g.graph.initializer)
  locks[name]=row
 assert json.loads((OUT/'trajectories/control/complete.json').read_text())['sampler_sha256']==json.loads((OUT/'trajectories/distill/complete.json').read_text())['sampler_sha256']
 freeze(OUT/'selection-lock.json',locks)
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis,rest,active=rig['gap_basis'],rig['rest']+rig['gap_offset'],rig['active']
 fresh=OUT/'confirmation/variants';older=SCALE/'datasets/confirmation';historical=APP/'.local/compact-gain-v2'
 reports={}
 for name,path in [('v4',BASELINE),('control',OUT/'control/model.onnx'),('distill',OUT/'distill/model.onnx')]:
  options=ort.SessionOptions();options.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
  cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True)
  report=dict(model_sha256=sha(path),bytes=path.stat().st_size)
  for cohort,root in [('fresh',fresh),('scale_regression',older),('historical',historical)]:
   rows=json.loads((root/'manifest.json').read_text())
   if cohort=='historical':rows=[r for r in rows if r['split']=='test']
   report[cohort]=evaluate_cohort(session,rows,root/'labels',cache,basis,rest,active)
  reports[name]=report;freeze(OUT/('evaluation-'+name+'.json'),report)
  print(name,report['fresh']['conditions']['original']['macro_gap_rmse_mm'],flush=True)
 def score(n,cohort='fresh',condition='original'):return reports[n][cohort]['conditions'][condition]['macro_gap_rmse_mm']
 def f1(n,cohort='fresh'):return reports[n][cohort]['conditions']['original']['macro_contact_f1']
 def closures(n):return {p:reports[n]['fresh']['conditions']['original']['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']}
 hits={n:closures(n) for n in reports}
 engines={e:1-reports['distill']['fresh']['conditions']['original']['engines'][e]['macro_gap_rmse_mm']/reports['control']['fresh']['conditions']['original']['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon','supertonic']}
 gain=lambda n:np.mean([score(n,condition=g) for g in ['half','boost']])
 checks=dict(primary_rms=1-score('distill')/score('control')>=.03,engines=min(engines.values())>=0,contact=f1('distill')>=f1('control'),gains=gain('distill')<=gain('control'),closures_total=sum(hits['distill'].values())>=sum(hits['control'].values()),closures_per_phone=min(hits['distill'][p]-hits['control'][p] for p in hits['distill'])>=-2)
 for cohort in ['historical','scale_regression']:
  checks[cohort+'_rms']=score('distill',cohort)<=1.02*score('v4',cohort);checks[cohort+'_f1']=f1('distill',cohort)>=f1('v4',cohort)-.01
 summary=dict(improvement_vs_control=1-score('distill')/score('control'),improvement_vs_v4=1-score('distill')/score('v4'),engine_improvement_vs_control=engines,well_timed_hits=hits,checks={k:bool(v) for k,v in checks.items()},promising=bool(all(checks.values())))
 freeze(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()

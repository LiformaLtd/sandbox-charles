"""Lock all three exports before scoring fresh combined/enriched/broad speech."""
from common import *
import numpy as np,onnx,onnxruntime as ort
from evaluate import evaluate_cohort

def main():
 ops=lambda g:[(n.op_type,list(n.input),list(n.output)) for n in g.graph.node];baseline=onnx.load(BASELINE);locks={}
 samples=[]
 for name in ['control','broad','targeted']:
  root=OUT/name;row=json.loads((root/'selection.json').read_text());assert sha(root/'model.onnx')==row['model_sha256'];assert row['bytes']==2251827 and row['parameters']==331047 and ops(onnx.load(root/'model.onnx'))==ops(baseline)
  locks[name]=row;samples.append(json.loads((OUT/'trajectories'/name/'complete.json').read_text()))
 assert len({s['old_sampler_sha256'] for s in samples})==1 and len({s['replay_sampler_sha256'] for s in samples})==1
 freeze(OUT/'selection-lock.json',locks)
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis,rest,active=rig['gap_basis'],rig['rest']+rig['gap_offset'],rig['active'];reports={}
 for name,path in [('v4',BASELINE)]+[(n,OUT/n/'model.onnx') for n in locks]:
  opt=ort.SessionOptions();opt.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider']);cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True);report=dict(model_sha256=sha(path),bytes=path.stat().st_size)
  for cohort,root in [('fresh',OUT/'datasets/confirmation'),('feature_regression',FEATURE/'confirmation/variants'),('historical',APP/'.local/compact-gain-v2')]:
   rows=json.loads((root/'manifest.json').read_text());rows=[r for r in rows if r['split']=='test'];report[cohort]=evaluate_cohort(session,rows,root/'labels',cache,basis,rest,active)
  reports[name]=report;freeze(OUT/('evaluation-'+name+'.json'),report);print(name,report['fresh']['conditions']['original']['macro_gap_rmse_mm'],flush=True)
 def score(n,cohort='fresh',gain='original'):return reports[n][cohort]['conditions'][gain]['macro_gap_rmse_mm']
 def f1(n,cohort='fresh'):return reports[n][cohort]['conditions']['original']['macro_contact_f1']
 hits={n:{p:r['fresh']['conditions']['original']['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']} for n,r in reports.items()}
 engine_gains={e:1-reports['targeted']['fresh']['conditions']['original']['engines'][e]['macro_gap_rmse_mm']/reports['broad']['fresh']['conditions']['original']['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon','supertonic']}
 broad_score=lambda n:reports[n]['fresh']['conditions']['original']['sources']['cmu-arctic-broad']['macro_gap_rmse_mm']
 gain_score=lambda n:np.mean([score(n,gain=g) for g in ['half','boost']])
 checks=dict(primary=1-score('targeted')/score('broad')>=.03,engines=min(engine_gains.values())>=0,contact=f1('targeted')>=f1('broad'),broad_stratum=broad_score('targeted')<=broad_score('broad'),gains=gain_score('targeted')<=gain_score('broad'),closure_total=sum(hits['targeted'].values())>=sum(hits['broad'].values()),closure_per_phone=min(hits['targeted'][p]-hits['broad'][p] for p in hits['targeted'])>=-2)
 for c in ['historical','feature_regression']:checks[c+'_rms']=score('targeted',c)<=1.02*score('v4',c);checks[c+'_f1']=f1('targeted',c)>=f1('v4',c)-.01
 summary=dict(improvement_vs_broad=1-score('targeted')/score('broad'),improvement_vs_control=1-score('targeted')/score('control'),improvement_vs_v4=1-score('targeted')/score('v4'),engine_improvement_vs_broad=engine_gains,hits=hits,checks={k:bool(v) for k,v in checks.items()},promising=bool(all(checks.values())))
 freeze(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()

"""Dense surface is primary; preserve gap/contact/timing and historical guard metrics."""
from common import *
import numpy as np,onnx,onnxruntime as ort
from evaluate import evaluate_cohort
from geometry import prepare,squared_error

def surface_metrics(rows,cache,rig):
 aggregates={};per_clip=[]
 for row in rows:
  with np.load(OUT/'surface-tracks'/(row['sha256_audio']+'.npz')) as z:stats=z['stats']
  pred=np.load(cache/(row['sha256_audio']+'.npy'))[:,rig['active']].astype(np.float64);errors=np.maximum(squared_error(pred,stats.astype(np.float64),rig['gram'].astype(np.float64)),0)
  key=row['condition'];voice=row['engine']+':'+row['voice'];r=aggregates.setdefault(key,{}).setdefault(voice,dict(sum=np.zeros(3),frames=0));r['sum']+=errors.sum(0);r['frames']+=len(errors)
  per_clip.append(dict(stem=row['stem'],condition=key,voice=voice,surface_rms_mm=float(10*np.sqrt(errors.sum(-1).mean()))))
 result={}
 for condition,voices in aggregates.items():
  detail={v:dict(surface_rms_mm=float(10*np.sqrt(r['sum'].sum()/r['frames'])),axis_rms_mm=(10*np.sqrt(r['sum']/r['frames'])).tolist(),frames=r['frames']) for v,r in voices.items()}
  engines={e:float(np.mean([r['surface_rms_mm'] for v,r in detail.items() if v.startswith(e+':')])) for e in ['phonon','supertonic']}
  result[condition]=dict(macro_surface_rms_mm=float(np.mean([r['surface_rms_mm'] for r in detail.values()])),macro_axis_rms_mm=np.mean([r['axis_rms_mm'] for r in detail.values()],axis=0).tolist(),engines=engines,voices=detail)
 return dict(conditions=result,clips=per_clip)
def main():
 locks={};ops=lambda g:[(n.op_type,list(n.input),list(n.output)) for n in g.graph.node];baseline=onnx.load(BASELINE)
 for name in ['control','surface']:
  root=OUT/name;r=json.loads((root/'selection.json').read_text());assert sha(root/'model.onnx')==r['model_sha256'];assert r['bytes']==2251827 and r['parameters']==331047 and ops(onnx.load(root/'model.onnx'))==ops(baseline);locks[name]=r
 assert json.loads((OUT/'trajectories/control/complete.json').read_text())['sampler_sha256']==json.loads((OUT/'trajectories/surface/complete.json').read_text())['sampler_sha256'];freeze(OUT/'selection-lock.json',locks)
 rig=prepare();gaprig=np.load(WORK/'arkit-student/rig-metrics.npz');reports={};fresh=OUT/'confirmation/variants';rows=json.loads((fresh/'manifest.json').read_text())
 for name,path in [('v4',BASELINE),('control',OUT/'control/model.onnx'),('surface',OUT/'surface/model.onnx')]:
  opt=ort.SessionOptions();opt.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider']);cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True);report=dict(model_sha256=sha(path),bytes=path.stat().st_size)
  for cohort,root in [('fresh',fresh),('feature_regression',FEATURE/'confirmation/variants'),('historical',APP/'.local/compact-gain-v2')]:
   subset=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='test'];report[cohort]=evaluate_cohort(session,subset,root/'labels',cache,gaprig['gap_basis'],gaprig['rest']+gaprig['gap_offset'],gaprig['active'])
  report['surface']=surface_metrics(rows,cache,rig);reports[name]=report;freeze(OUT/('evaluation-'+name+'.json'),report);print(name,'surface',report['surface']['conditions']['original']['macro_surface_rms_mm'],'gap',report['fresh']['conditions']['original']['macro_gap_rmse_mm'],flush=True)
 # The original fitted coefficients give a reference retargeting residual, not a strict oracle.
 oracle_cache=OUT/'fitted-reference';oracle_cache.mkdir(exist_ok=True)
 for row in rows:
  with np.load(fresh/'labels'/(row['stem']+'.npz')) as z:weights=z['weights']
  p=oracle_cache/(row['sha256_audio']+'.npy')
  if not p.exists():np.save(p,weights)
 freeze(OUT/'fitted-surface-reference.json',surface_metrics(rows,oracle_cache,rig))
 gap=lambda n,c='fresh':reports[n][c]['conditions']['original']['macro_gap_rmse_mm']
 f1=lambda n,c='fresh':reports[n][c]['conditions']['original']['macro_contact_f1']
 surf=lambda n,g='original':reports[n]['surface']['conditions'][g]['macro_surface_rms_mm']
 hits={n:{p:r['fresh']['conditions']['original']['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']} for n,r in reports.items()}
 gains={e:1-reports['surface']['surface']['conditions']['original']['engines'][e]/reports['control']['surface']['conditions']['original']['engines'][e] for e in ['phonon','supertonic']}
 gap_engine={e:1-reports['surface']['fresh']['conditions']['original']['engines'][e]['macro_gap_rmse_mm']/reports['control']['fresh']['conditions']['original']['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon','supertonic']}
 checks=dict(surface_primary=1-surf('surface')/surf('control')>=.03,surface_engines=min(gains.values())>=0,gap=gap('surface')<=1.01*gap('control'),gap_engines=min(gap_engine.values())>=-.01,contact=f1('surface')>=f1('control')-.002,closures_total=sum(hits['surface'].values())>=sum(hits['control'].values()),closures_phone=min(hits['surface'][p]-hits['control'][p] for p in hits['surface'])>=-2,gain_surface=np.mean([surf('surface',g) for g in ['half','boost']])<=np.mean([surf('control',g) for g in ['half','boost']]))
 for c in ['historical','feature_regression']:checks[c+'_gap']=gap('surface',c)<=1.02*gap('v4',c);checks[c+'_f1']=f1('surface',c)>=f1('v4',c)-.01
 summary=dict(surface_improvement_vs_control=1-surf('surface')/surf('control'),surface_improvement_vs_v4=1-surf('surface')/surf('v4'),gap_improvement_vs_control=1-gap('surface')/gap('control'),surface_engine_gains=gains,gap_engine_gains=gap_engine,hits=hits,checks={k:bool(v) for k,v in checks.items()},promising=bool(all(checks.values())))
 freeze(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()

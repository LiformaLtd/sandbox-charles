"""Supplementary summaries after locked primary scoring; never checkpoint selection."""
from common import *
import numpy as np,onnx
from onnx import numpy_helper
sys.path.insert(0,str(APP/'scripts/research/lip_surface'))
from geometry import prepare,squared_error

def main():
 reports={a:json.loads((OUT/f'evaluation-{a}.json').read_text()) for a in ['v4','control','balanced']};result=dict(source_sha256=sha(__file__),note='Supplementary bootstrap and dense-surface regression summaries; not selection or a new fresh surface benchmark.')
 rows={a:[r for r in reports[a]['fresh']['clips'] if r['condition']=='original'] for a in reports};texts=sorted({r['text'] for r in rows['v4']});voices=sorted({r['engine']+':'+r['voice'] for r in rows['v4']});assert len(texts)==24 and len(voices)==4
 pick=np.random.default_rng(837).integers(24,size=(2000,24));boot={}
 for arm in reports:
  matrix=np.empty((4,24,2))
  for r in rows[arm]:matrix[voices.index(r['engine']+':'+r['voice']),texts.index(r['text'])]=[r['squared_gap_error'],r['gap_values']]
  totals=matrix[:,pick].sum(2);boot[arm]=10*np.sqrt(totals[:,:,0]/totals[:,:,1]).mean(0)
 result['prompt_cluster_bootstrap']=dict(balanced_improvement_vs_control_ci95=np.quantile(1-boot['balanced']/boot['control'],[.025,.975]).tolist(),replicates=2000,seed=837,unit='24 transcript clusters, retaining four voices; conditional on chosen presets and this cohort')
 rig=prepare();surface=APP/'.local/compact-surface-v1';cases=json.loads((surface/'confirmation/variants/manifest.json').read_text());dense={}
 for arm in reports:
  groups={};cache=OUT/'predictions'/reports[arm]['model_sha256']
  for row in cases:
   if row['condition']!='original':continue
   with np.load(surface/'surface-tracks'/(row['sha256_audio']+'.npz')) as z:stats=z['stats'].astype(np.float64)
   pred=np.load(cache/(row['sha256_audio']+'.npy'))[:,rig['active']].astype(np.float64);errors=np.maximum(squared_error(pred,stats,rig['gram'].astype(np.float64)),0);voice=row['engine']+':'+row['voice'];r=groups.setdefault(voice,np.zeros(4));r[:3]+=errors.sum(0);r[3]+=len(errors)
  byvoice={v:10*np.sqrt(r[:3].sum()/r[3]) for v,r in groups.items()};dense[arm]=dict(macro_surface_rms_mm=float(np.mean(list(byvoice.values()))),voices=byvoice)
 result['surface_regression']=dense
 browser={}
 for arm in reports:
  medians=[];memory=[]
  for run in range(1,4):
   entries=json.loads((OUT/f'browser/parity/run-{run}.json').read_text());subset=[r for r in entries if r['model']==arm];medians.append(float(np.median([r['median_ms'] for r in subset if 'packed' in r['name']])));memory.append(json.loads((OUT/f'memory/run-{run}/{arm}.json').read_text())['peak']/2**20)
  startup=[r['session_create_ms'] for r in json.loads((OUT/'browser/startup.json').read_text())['rows'] if r['model']==arm]
  browser[arm]=dict(long_clip_block_median_ms_trials=medians,median_ms=float(np.median(medians)),nominal_500ms_realtime=500/np.median(medians),memory_peak_mib_trials=memory,explicit_gpu_bytes=subset[0]['allocation']['gpuBytes'],startup_median_ms=float(np.median(startup)))
 result['browser']=browser
 base={v.name:numpy_helper.to_array(v) for v in onnx.load(BASELINE).graph.initializer};unchanged={}
 for arm in ['control','balanced']:
  graph={v.name:numpy_helper.to_array(v) for v in onnx.load(OUT/arm/'model.onnx').graph.initializer}
  names=['filters','mel','student.mean','student.scale','student.left','student.fraction','student.channel_map']
  for k in names:np.testing.assert_array_equal(base[k],graph[k])
  unchanged[arm]=names
 result['unchanged_export_tensors']=unchanged
 freeze(OUT/'analysis.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

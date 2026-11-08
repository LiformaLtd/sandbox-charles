"""Aggregate all-gain/original training metrics without mixing their populations."""
from common import *
import numpy as np
from evaluate import aggregate

def main():
 reports={a:OUT/f'evaluation-{a}.json' for a in MODELS};reports['head-refit']=OUT/'head-evaluation/evaluation-head-refit.json';summary={}
 for arm,path in reports.items():
  r=json.loads(path.read_text());clips=sum([r['cohorts'][part]['clips'] for part in ['old-train','new-train']],[]);conditions={}
  for condition in ['all','original','half','boost']:
   selected=clips if condition=='all' else [c for c in clips if c['condition']==condition];stats={}
   for part in ['old-train','new-train']:
    for v in r['fit_statistics'][part]:
     if condition!='all' and v['condition']!=condition:continue
     stats.setdefault(v['voice'],np.zeros(8));stats[v['voice']]+=v['sums']
   conditions[condition]=dict(**aggregate(selected),reference_discrepancy_mm=float(np.mean([10*np.sqrt(v[2]/v[1]) for v in stats.values()])),student_vs_fitted_gap_mm=float(np.mean([10*np.sqrt(v[0]/v[1]) for v in stats.values()])),frames=int(sum(v[6] for v in stats.values())))
  summary[arm]=dict(model_sha256=r['model_sha256'],report_sha256=sha(path),primary_train=conditions,test_original=r['fresh']['original'],parity_cases=len(r['native_parity']))
 freeze(OUT/'summary.json',dict(models=summary,source_sha256=sha(__file__),note='All-primary training3840 clips; original-gain training1280; test96 original clips. Head test is previously inspected regression evidence. Reference discrepancy is not a strict optimum.'))
 for arm,r in summary.items():print(arm,'all-primary',r['primary_train']['all']['macro_gap_rmse_mm'],'original-primary',r['primary_train']['original']['macro_gap_rmse_mm'],'test',r['test_original']['macro_gap_rmse_mm'])
if __name__=='__main__':main()

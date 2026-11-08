"""Combine complete fit measurements with the independent test screen."""
from common import *
import numpy as np
from evaluate import aggregate

def summarize(report):
 clips=sum([report['cohorts'][part]['clips'] for part in ['old-train','new-train']],[])
 return {c:aggregate(clips if c=='all' else [r for r in clips if r['condition']==c]) for c in ['all','original','half','boost']}
def main():
 paths={'v4':APP/'.local/compact-fit-v1/evaluation-v4.json','start':APP/'.local/compact-fit-v1/evaluation-control.json',**{a:OUT/'fit'/f'evaluation-{a}.json' for a in ['control','protected']}}
 reports={a:json.loads(p.read_text()) for a,p in paths.items()}
 for a,r in reports.items():assert r['model_sha256']==json.loads((OUT/f'evaluation-{a}.json').read_text())['model_sha256']
 scores={a:summarize(r) for a,r in reports.items()};gain=1-scores['protected']['original']['macro_gap_rmse_mm']/scores['control']['original']['macro_gap_rmse_mm'];test=json.loads((OUT/'summary.json').read_text());checks=dict(test['checks'],training_rms=gain>=.03)
 freeze(OUT/'fit-summary.json',dict(models=scores,source_sha256=sha(__file__),report_sha256={a:sha(p) for a,p in paths.items()},training_improvement_vs_control=gain,checks=checks,promising=bool(all(checks.values()))))
 print((OUT/'fit-summary.json').read_text(),flush=True)
if __name__=='__main__':main()

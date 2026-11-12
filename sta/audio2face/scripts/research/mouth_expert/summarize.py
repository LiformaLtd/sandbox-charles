"""Apply predeclared paired corruption gates; expose each failure and engine."""
from common import *
import numpy as np
REQUIRED=['shift-m8','shift-m4','shift-p4','shift-p8','underopen-50','missing-closure']

def summarize(rows,value):
 result={}
 for mode in sorted({r['mode'] for r in rows}):
  selected=[r for r in rows if r['mode']==mode and r.get('split','test')=='test'];base={r['case']:r[value] for r in selected if r['kind']=='correct'};groups={}
  for kind in sorted({r['kind'] for r in selected if r['kind']!='correct'}):
   cases=[dict(case=r['case'],engine=r['engine'],delta=r[value]-base[r['case']],correct=base[r['case']],corrupted=r[value]) for r in selected if r['kind']==kind];wins=sum(r['delta']>1e-5 for r in cases);engines={e:sum(r['delta']>1e-5 for r in cases if r['engine']==e) for e in ['phonon','supertonic']};groups[kind]=dict(wins=wins,n=len(cases),mean_delta=float(np.mean([r['delta'] for r in cases])),median_delta=float(np.median([r['delta'] for r in cases])),engine_wins=engines,passes=bool(wins>=6 and min(engines.values())>=2),cases=cases)
  result[mode]=dict(passes=bool(all(groups[k]['passes'] for k in REQUIRED)),comparisons=groups)
 return result

def main():
 av=summarize(json.loads((OUT/'av-scores.json').read_text()),'loss');sync=summarize(json.loads((OUT/'sync-scores.json').read_text()),'distance');controls=json.loads((OUT/'controls.json').read_text());result=dict(av=av,sync=sync,required=REQUIRED,controls=dict(max_repeat_delta=max(r['repeat_delta'] for r in controls),correct_transcript_preferred=sum(r['correct']<r['reversed_token_loss'] for r in controls),n=len(controls)),source_sha256=sha(__file__))
 freeze(OUT/'summary.json',result)
 for family,groups in [('AV',av),('Sync',sync)]:
  for mode,r in groups.items():print(family,mode,'PASS',r['passes'],{k:v['wins'] for k,v in r['comparisons'].items()},flush=True)
 print('Controls',result['controls'],flush=True)
if __name__=='__main__':main()

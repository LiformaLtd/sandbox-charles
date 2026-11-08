"""Supplementary exact MSE decomposition for the constructed warm-history mixture."""
import numpy as np
from common import *
from geometry import prepare

def main():
 cases=json.loads((OUT/'cases.json').read_text());rig=prepare();gaprig=np.load(WORK/'arkit-student/rig-metrics.npz');time=np.arange(225)/60;rows=[]
 for case in cases:
  surface=[];gaps=[]
  for history in ['silence','speech-a','speech-b']:
   with np.load(OUT/'tracks'/case['id']/(history+'-s0.npz')) as z:surface.append(z['surface'].astype(np.float64));gaps.append(z['gaps'].astype(np.float64))
  surface=np.stack(surface);gaps=np.stack(gaps);w=np.load(OUT/'tracks'/case['id']/'student.npy')[:,rig['active']].astype(np.float64)
  p=np.einsum('tc,cva->tva',w,rig['basis'],optimize=False)-rig['offset'];g=np.einsum('tc,ck->tk',w,gaprig['gap_basis'][rig['active']],optimize=False)+gaprig['rest']+gaprig['gap_offset']
  for lo,hi in [[.25,.75],[.75,1.25],[1.25,2.25],[2.25,3.25],[.25,3.25]]:
   take=(time>=lo)&(time<hi)
   for name,target,pred in [('surface',surface[:,take],p[take]),('gaps',gaps[:,take],g[take])]:
    mean=target.mean(0);norm=lambda a:np.square(a).sum(-1).mean() if name=='surface' else np.square(a).mean()
    variance=float(norm(target-mean[None]));bias=float(norm(pred-mean));error=float(norm(pred[None]-target));assert abs(error-variance-bias)<1e-10
    rows.append(dict(case=case['id'],metric=name,bin=[lo,hi],variance_cm2=variance,bias_cm2=bias,student_mse_cm2=error))
 groups={}
 for lo,hi in [[.25,.75],[.75,1.25],[1.25,2.25],[2.25,3.25],[.25,3.25]]:
  group={}
  for metric in ['surface','gaps']:
   selected=[r for r in rows if r['metric']==metric and r['bin']==[lo,hi]];v,b,e=[float(np.mean([r[k] for r in selected])) for k in ['variance_cm2','bias_cm2','student_mse_cm2']]
   group[metric]=dict(conditional_floor_rms_mm=10*np.sqrt(v),student_error_uniform_warm_histories_mm=10*np.sqrt(e),history_variance_fraction_of_probe_student_mse=v/e,conditional_mean_bias_rms_mm=10*np.sqrt(b))
  groups[f'{lo}-{hi}']=group
 freeze(OUT/'mse-decomposition.json',dict(groups=groups,rows=rows,source_sha256=sha(__file__),note='Supplementary analysis; variance fraction applies only to this constructed uniform warm-history mixture, not production speech or expected recurrent-model gain.'))
 print(json.dumps(groups,indent=2))
if __name__=='__main__':main()

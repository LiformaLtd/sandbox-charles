"""Recheck all frozen inputs and validate frame/corruption coverage."""
from common import *
import numpy as np,cv2

def main():
 protocol=json.loads((OUT/'protocol.json').read_text())
 for name,digest in protocol['sources'].items():assert sha(APP/name)==digest,name
 for name,digest in protocol['weights'].items():assert sha(VENDOR/name)==digest,name
 for name,digest in protocol['case_files_sha256'].items():assert sha(OUT/name)==digest,name
 for name,digest in protocol['rig_sha256'].items():assert sha(APP/'.local'/name)==digest,name
 assert sha(OUT/'cases.json')==protocol['cases_sha256'];names=json.loads((WORK/'arkit-student/channels.json').read_text());jaw=names.index('jawOpen');rig=np.load(WORK/'arkit-student/rig-metrics.npz');B=rig['gap_basis'];rest=rig['rest']+rig['gap_offset'];rows=[]
 prior={r['text'] for r in json.loads((OUT/'cases.json').read_text())}
 for root in [OUT,OUT/'confirmation']:
  cases=json.loads((root/'cases.json').read_text())
  if root!=OUT:
   p=json.loads((root/'protocol.json').read_text());assert p['global_offset_frames']==-1 and sha(OUT/'sync-calibration.json')==p['sync_calibration_sha256']
   for n,d in p['files'].items():assert sha(root/n)==d
   for n,d in p['sources'].items():assert sha(APP/n)==d
   assert not any(r['text'] in prior for r in cases)
  for case in cases:
   n=case['frames'];tracks={k:np.fromfile(root/'tracks'/(case['id']+'-'+k+'.f32'),'<f4').reshape(n,52) for k in ['correct','underopen-50','missing-closure']}
   for k,v in tracks.items():
    assert np.isfinite(v).all() and v.min()>=-1e-6 and v.max()<=1.000001
    if k!='correct':np.testing.assert_array_equal(np.delete(v,jaw,axis=1),np.delete(tracks['correct'],jaw,axis=1))
    for view in ['mouth','face']:
     files=list((root/'frames'/case['id']/k/view).glob('*.png'));assert len(files)==n
     assert all(cv2.imread(str(p)).shape==(224,224,3) for p in files)
   g=np.einsum('tc,ck->tk',tracks['correct'],B,optimize=False)+rest;closed=np.median(g,-1)<=.1;np.testing.assert_array_equal(tracks['underopen-50'][closed],tracks['correct'][closed]);changed=np.any(tracks['missing-closure']!=tracks['correct'],axis=-1);windows=np.arange(18,n-5-18)-1;affected=sum(changed[t:t+5].any() for t in windows)
   if case.get('split')!='calibration':assert affected>0
   rows.append(dict(case=case['id'],sync_windows=len(windows),closure_corruption_windows=int(affected)))
 freeze(OUT/'input-audit.json',dict(cases=rows,all_inputs_verified=True,confirmation_texts_disjoint=True,source_sha256=sha(__file__)));print('Audited20 cases, all frame dimensions/corruptions and frozen inputs.',flush=True)
if __name__=='__main__':main()

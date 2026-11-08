from common import *
import numpy as np,cv2
from score import Sync,KINDS,movie
from summarize import summarize
CONFIRM=OUT/'confirmation'
def main():
 protocol=json.loads((CONFIRM/'protocol.json').read_text())
 for name,d in protocol['sources'].items():assert sha(APP/name)==d
 for name,d in protocol['files'].items():assert sha(CONFIRM/name)==d
 assert sha(OUT/'sync-calibration.json')==protocol['sync_calibration_sha256'];cases=json.loads((CONFIRM/'cases.json').read_text());sync=Sync();rows=[]
 for case in cases:
  n=case['frames'];pcm=np.fromfile(CONFIRM/'audio'/(case['id']+'.f32'),'<f4')
  def images(kind):return np.stack([cv2.cvtColor(cv2.imread(str(CONFIRM/'frames'/case['id']/kind/'face'/f'{t:04}.png')),cv2.COLOR_BGR2RGB) for t in range(n)])
  base=images('correct');v,a=sync.features(base,pcm);features={'correct':v}
  for kind in ['underopen-50','missing-closure']:features[kind]=sync.features(images(kind),pcm)[0]
  features['static']=sync.features(movie(base,'static'),pcm)[0];t=np.arange(18,len(a)-18)
  for kind in KINDS:
   delta=(-1 if kind[6]=='m' else 1)*int(kind[7:]) if kind.startswith('shift-') else 0;key='correct' if kind.startswith('shift-') else kind;score=float(np.linalg.norm(features[key][t-1+delta]-a[t],axis=-1).mean());rows.append(dict(case=case['id'],engine=case['engine'],voice=case['voice'],kind=kind,mode='fixed-confirmation',distance=score,windows=len(t)))
  print(case['id'],'scored',flush=True)
 freeze(CONFIRM/'scores.json',rows);result=summarize(rows,'distance');freeze(CONFIRM/'summary.json',result);print(json.dumps({k:{'passes':r['passes'],'wins':{a:b['wins'] for a,b in r['comparisons'].items()}} for k,r in result.items()},indent=2))
if __name__=='__main__':main()

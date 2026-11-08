"""Regenerate compact exact dense-surface targets, checking existing nine-gap references."""
import argparse,time,numpy as np
from common import *
from teacher import backend,animate,MODEL_SHA256

def recipe():
 return dict(teacher_sha256=MODEL_SHA256,identity='James',seed=0,tf32=False,projector_sha256=sha(HERE/'teacher.py'),geometry_sha256=sha(HERE/'geometry.py'),assets={n:sha(ASSETS/n) for n in ['model_data_James.npz','bs_skin_James.npz','bs_skin_config_James.json']},teacher_source={p.name:sha(p) for p in (TEACHER/'app/audio2face_v3').glob('*.py')})
def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['primary','confirmation'],required=True);args=p.parse_args()
 meta=recipe();signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();freeze(OUT/'surface-recipe.json',meta)
 roots=[SCALE/'datasets'/n for n in ['old-train','new-train','common-dev']] if args.cohort=='primary' else [OUT/'confirmation/variants']
 records=[(row,root/'labels'/(row['stem']+'.npz')) for root in roots for row in json.loads((root/'manifest.json').read_text())]
 folder=OUT/'surface-tracks';folder.mkdir(exist_ok=True);run=None;began=time.monotonic();max_gap_delta=0.;lock={}
 for i,(row,reference) in enumerate(records):
  assert sha(row['audio'])==row['sha256_audio'];key=row['sha256_audio'];path=folder/(key+'.npz')
  with np.load(reference) as z:
   truth=z['gaps'];assert str(z['audio_sha256'])==key
  if not path.exists():
   if run is None:run=backend()
   pcm=np.fromfile(row['audio'],'<f4');parts=[];cursor=0
   for indices,values in animate(pcm,run,identity='James',seed=0):
    np.testing.assert_array_equal(indices,np.arange(cursor,cursor+len(indices))*16000//60);parts.append(values);cursor+=len(indices)
   values=np.concatenate(parts);assert values.shape==(len(truth),129) and np.isfinite(values).all()
   np.testing.assert_allclose(values[:,120:],truth,atol=2e-4,rtol=1e-4)
   temp=path.with_suffix('.part.npz');np.savez_compressed(temp,stats=values[:,:120].astype(np.float32),gaps=values[:,120:],recipe_sha256=signature,audio_sha256=key);temp.replace(path)
  with np.load(path) as z:
   assert str(z['recipe_sha256'])==signature and str(z['audio_sha256'])==key and z['stats'].shape==(len(truth),120)
   delta=float(np.max(abs(z['gaps']-truth)));np.testing.assert_allclose(z['gaps'],truth,atol=2e-4,rtol=1e-4);max_gap_delta=max(max_gap_delta,delta)
  lock[row['stem']]=dict(audio_sha256=key,reference_sha256=sha(reference),surface_sha256=sha(path))
  if (i+1)%50==0 or i==0:print(args.cohort,i+1,'/',len(records),'elapsed',round(time.monotonic()-began),flush=True)
 freeze(OUT/('surface-'+args.cohort+'-lock.json'),dict(records=lock,recipe_sha256=signature,max_gap_delta=max_gap_delta,source_sha256=sha(__file__),seconds=time.monotonic()-began))
 print(args.cohort,'complete; max gap delta',max_gap_delta,flush=True)
if __name__=='__main__':main()

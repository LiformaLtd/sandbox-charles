"""Map the already aligned teacher speech tracks onto exactly the cached primary windows."""
import numpy as np
from common import *
def main():
 for name in ['old-train','new-train']:
  root=SCALE/'datasets'/name;rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='train']
  cache=SCALE/'features'/name;n=json.loads((cache/'complete.json').read_text())['windows'];dest=OUT/'hints'/name;done=dest/'complete.json'
  if done.exists():continue
  dest.mkdir(parents=True,exist_ok=True);hints=np.lib.format.open_memmap(dest/'hints.part.npy',mode='w+',dtype=np.float16,shape=(n,30,768))
  expected=np.load(cache/'weights.npy',mmap_mode='r');masks=np.load(cache/'valid.npy',mmap_mode='r');active=np.load(WORK/'arkit-student/rig-metrics.npz')['active'];cursor=0;fingerprints={}
  for row in rows:
   p=root/'labels'/(row['stem']+'.npz');fingerprints[row['stem']]=sha(p)
   if sha(row['audio'])!=row['sha256_audio']:raise ValueError('Changed waveform')
   with np.load(p) as z:
    features=z['speech_features'];weights=z['weights'][:,active]
    if features.shape!=(len(weights),768) or not np.isfinite(features).all() or str(z['audio_sha256'])!=row['sha256_audio']:raise ValueError('Invalid hints')
   position=0;offsets=np.arange(15,45)*16000//60
   for start in range(-8000,row['samples']-4000,8000):
    valid=(start+offsets>=0)&(start+offsets<row['samples']);count=int(valid.sum())
    np.testing.assert_array_equal(valid,masks[cursor]);np.testing.assert_array_equal(weights[position:position+count],expected[cursor,valid])
    hints[cursor]=0;hints[cursor,valid]=features[position:position+count];position+=count;cursor+=1
   if position!=len(features):raise ValueError('Incomplete feature timeline')
  assert cursor==n;hints.flush();del hints;(dest/'hints.part.npy').replace(dest/'hints.npy')
  freeze(done,dict(windows=n,source_sha256=sha(__file__),manifest_sha256=sha(root/'manifest.json'),label_sha256=fingerprints,hints_sha256=sha(dest/'hints.npy')))
  print(name,n,'aligned hint windows checked against geometry/masks',flush=True)
if __name__=='__main__':main()

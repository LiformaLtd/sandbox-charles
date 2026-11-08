"""Stream aligned float32 sufficient-statistic maps while teacher tracks arrive."""
import time,numpy as np
from common import *
def main():
 for name in ['old-train','new-train','common-dev']:
  root=SCALE/'datasets'/name;rows=json.loads((root/'manifest.json').read_text());cache=SCALE/'features'/name;dest=OUT/'surface-maps'/name;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'complete.json').exists():continue
  valid=np.load(cache/'valid.npy',mmap_mode='r');n=len(valid);array=np.lib.format.open_memmap(dest/'stats.part.npy',mode='w+',dtype=np.float32,shape=(n,30,120));cursor=0;fingerprints={}
  for row in rows:
   path=OUT/'surface-tracks'/(row['sha256_audio']+'.npz')
   while not path.exists():time.sleep(1)
   with np.load(path) as z:
    stats=z['stats'];assert str(z['audio_sha256'])==row['sha256_audio']
   frames=(row['samples']*60+15999)//16000;assert stats.shape==(frames,120);position=0
   for start in range(-8000,row['samples']-4000,8000):
    mask=(start+np.arange(15,45)*16000//60>=0)&(start+np.arange(15,45)*16000//60<row['samples']);np.testing.assert_array_equal(mask,valid[cursor]);count=int(mask.sum())
    array[cursor]=0;array[cursor,mask]=stats[position:position+count];cursor+=1;position+=count
   assert position==frames;fingerprints[row['stem']]=sha(path)
  assert cursor==n;array.flush();del array;(dest/'stats.part.npy').replace(dest/'stats.npy')
  freeze(dest/'complete.json',dict(windows=n,track_sha256=fingerprints,manifest_sha256=sha(root/'manifest.json'),array_sha256=sha(dest/'stats.npy'),source_sha256=sha(__file__)))
  print(name,n,'surface windows complete',flush=True)
if __name__=='__main__':main()

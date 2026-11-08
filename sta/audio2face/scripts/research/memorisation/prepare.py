"""Select tiny training-only set by hash, before measuring errors."""
from common import *
import numpy as np
from features import Dataset

def main():
 assert sha(BASELINE)==EXPECTED
 root=SCALE/'datasets/old-train';rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='train'];voices=sorted({r['engine']+':'+r['voice'] for r in rows});assert len(voices)==8
 selected=[];seen=set()
 for voice in voices:
  eligible=[r for r in rows if r['engine']+':'+r['voice']==voice and r['condition']=='original']
  eligible.sort(key=lambda r:hashlib.sha256(('memorisation-v4-73:'+r['text']).encode()).hexdigest())
  pick=next(r for r in eligible if r['text'] not in seen);seen.add(pick['text']);selected.append(pick)
 cursor=0;indices=[];locks={};metadata=[]
 for row in rows:
  count=len(range(-8000,Path(row['audio']).stat().st_size//4-4000,8000))
  if row in selected:
   indices.extend(range(cursor,cursor+count));label=root/'labels'/(row['stem']+'.npz');locks[str(row['audio'])]=sha(row['audio']);locks[str(label)]=sha(label);metadata.append(dict(row,windows=count))
  cursor+=count
 dataset=Dataset(['old-train']);assert cursor==dataset.length
 OUT.mkdir(exist_ok=True);freeze(OUT/'manifest.json',metadata);np.savez(OUT/'batch.npz',**dataset.batch(indices),indices=np.array(indices))
 files=[HERE/n for n in ['common.py','prepare.py','train.py','export.py']]+[APP/'scripts/research/data_scale/mlx_backend.py',APP/'scripts/research/data_scale/features.py',APP/'docs/plans/v4-memorisation.md']
 freeze(OUT/'protocol.json',dict(baseline=str(BASELINE),baseline_sha256=EXPECTED,steps=3000,seed=37,clips=8,voices=voices,windows=len(indices),batch_sha256=sha(OUT/'batch.npz'),manifest_sha256=sha(OUT/'manifest.json'),source_sha256={str(p.relative_to(APP)):sha(p) for p in files},inputs_sha256=locks,training_only=True))
 for p in files:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 freeze(OUT/'sources/model.py',(SCALE/'sources/model.py').read_bytes());print('Frozen eight training clips /',len(indices),'windows.',flush=True)
if __name__=='__main__':main()

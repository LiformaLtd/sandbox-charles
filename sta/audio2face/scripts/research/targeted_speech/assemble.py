"""Verify independent teacher labels and assemble novel training/test views."""
import numpy as np
from common import *
def main():
 if not(OUT/'corpus/complete.json').exists():raise ValueError('Corpus not complete')
 sets={a:[] for a in ['broad','targeted','confirmation']};labels={};recipes=[]
 for root in sorted((OUT/'corpus/label-shards').iterdir()):
  recipes.append(json.loads((root/'label-recipe.json').read_text()))
  for row in json.loads((root/'manifest.json').read_text()):
   p=root/'labels'/(row['stem']+'.npz');assert sha(row['audio'])==row['sha256_audio']
   with np.load(p) as z:
    frames=(row['samples']*60+15999)//16000;assert z['weights'].shape==(frames,52) and z['gaps'].shape==(frames,9) and str(z['audio_sha256'])==row['sha256_audio']
   labels[row['stem']]=p;sets[row['arm']].append(row)
 assert all(r==recipes[0] for r in recipes) and [len(sets[a]) for a in sets]==[288,288,512]
 for name,rows in sets.items():
  root=OUT/'datasets'/name;(root/'labels').mkdir(parents=True,exist_ok=True);freeze(root/'manifest.json',rows)
  for row in rows:
   p=root/'labels'/(row['stem']+'.npz')
   if not p.exists():p.symlink_to(labels[row['stem']])
 freeze(OUT/'assembly-lock.json',dict(manifests={n:sha(OUT/'datasets'/n/'manifest.json') for n in sets},teacher_recipe=recipes[0],source_sha256=sha(__file__)))
 print('288 broad +288 targeted training labels;512 fresh cases verified.',flush=True)
if __name__=='__main__':main()

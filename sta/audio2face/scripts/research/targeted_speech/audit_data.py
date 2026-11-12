"""Check actual frozen corpus separation, pair assignments and voice/gain counts."""
from collections import Counter
from common import *
import re

def main():
 root=OUT/'corpus';protocol=json.loads((root/'protocol.json').read_text());assert sha(root/'planned.json')==protocol['planned_sha256']
 norm=lambda t:' '.join(re.findall('[a-z]+',t.lower()))
 rows=json.loads((root/'planned.json').read_text());texts={arm:{norm(r['text']) for r in rows if r['arm']==arm} for arm in ['broad','targeted','confirmation']}
 assert list(map(len,texts.values()))==[72,72,32]
 assert not(texts['broad']&texts['targeted'] or texts['broad']&texts['confirmation'] or texts['targeted']&texts['confirmation'])
 audit=json.loads((root/'audit.json').read_text());prior=set()
 for name,digest in audit['prior_manifests'].items():
  p=Path(name);assert sha(p)==digest,name;prior.update(norm(r['text']) for r in json.loads(p.read_text()) if isinstance(r,dict) and 'text' in r)
 assert not set.union(*texts.values())&prior
 for arm in ['broad','targeted']:
  group=[r for r in rows if r['arm']==arm];c=Counter((r['engine'],r['voice'],r['training_gain']) for r in group)
  assert len(c)==24 and set(c.values())=={12}
  peer={r['stem'].rsplit('-',1)[-1]+':'+r['engine']+':'+r['voice']:r['training_gain'] for r in group}
  if arm=='broad':broad=peer
  else:assert peer==broad
 assert audit['targeted_mean_score']>audit['broad_mean_score']
 print('Disjoint texts, unchanged prior inventory, paired voices and exact gain balance passed.',flush=True)
if __name__=='__main__':main()

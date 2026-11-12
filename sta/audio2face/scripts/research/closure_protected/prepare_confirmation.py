"""Freeze fresh transcripts and pinned synthesis inputs before evaluating any model."""
import json,re,shutil
from common import *
TEXTS=[
 ('closure-m','Mum might make more marmalade tomorrow morning.'),
 ('closure-m','Mandy made a warm meal for the men in the mill.'),
 ('closure-m','The monkey moved among the many maple trees.'),
 ('closure-b','Bobby brought a bowl of black beans to the table.'),
 ('closure-b','Betty bought a beautiful blue bag for her brother.'),
 ('closure-b','The boy blew bubbles beside the broken bridge.'),
 ('closure-p','Pat put a pair of purple slippers beside the piano.'),
 ('closure-p','Penny painted a picture of a puppy playing.'),
 ('closure-p','Please pass the pepper and put the pan away.'),
 ('vowels','The little girl watched the wide ocean from the shore.'),
 ('vowels','We rode through green fields under a clear blue sky.'),
 ('vowels','He heard a quiet voice coming from the old house.'),
]

def main():
 norm=lambda t:' '.join(re.findall('[a-z]+',t.lower()))
 prior=set();sources={}
 normal=Path('/Users/charleswiles/Source/liforma/sandboxes/sandbox-charles/sta/audio2face/.local')
 paths=list(normal.rglob('manifest.json'))+list(normal.rglob('planned.json'))
 paths += [WORK/p/'manifest.json' for p in ['arkit-student','arkit-student-v2','arkit-student-v3','arkit-student-v3/extra']]
 paths += [APP/'.local/compact-syncnet-v1/confirmation/base/planned.json',APP/'.local/compact-loss-v1/confirmation/base/planned.json',APP/'.local/mouth-expert-v1/cases.json',APP/'.local/mouth-expert-v1/confirmation/cases.json',APP/'.local/compact-recurrent-v1/confirmation/base/planned.json',APP/'.local/compact-surface-v1/confirmation/base/planned.json',SCALE/'corpus/planned.json',TARGETED/'corpus/planned.json',FEATURE/'confirmation/base/planned.json',APP/'.local/compact-temporal-v1/confirmation/base/planned.json']
 for p in sorted(set(paths)):
  try:rows=json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):continue
  if not isinstance(rows,list):continue
  texts=[r['text'] for r in rows if isinstance(r,dict) and isinstance(r.get('text'),str)]
  if texts:prior.update(map(norm,texts));sources[str(p)]=sha(p)
 pilot=APP/'.local/compact-coverage-v2/base';lex=json.loads((pilot/'assets/aligner/lexicon.json').read_text())['lexicon']
 supported=lambda t:all(w.upper() in lex for w in re.findall("[A-Za-z]+(?:'[A-Za-z]+)*",t))
 prompts=[dict(kind=k,text=t,source='authored') for k,t in TEXTS]
 for r in prompts:
  if norm(r['text']) in prior or not supported(r['text']):raise ValueError('Repeated or unsupported: '+r['text'])
 arctic=[]
 for line in (SCALE/'text-sources/cmuarctic.data').read_text().splitlines():
  m=re.fullmatch(r'\(\s*(\S+)\s+"(.*)"\s*\)',line)
  if m and norm(m[2]) not in prior and supported(m[2]) and 5<=len(m[2].split())<=22:arctic.append(dict(kind='broad-speech',text=m[2],source='cmu-arctic',source_id=m[1]))
 arctic.sort(key=lambda r:__import__('hashlib').sha256(('closure-protected-37:'+r['text']).encode()).hexdigest())
 if len(arctic)<12:raise ValueError('Insufficient unseen broad prompts')
 prompts+=arctic[:12];assert len({norm(r['text']) for r in prompts})==24
 assets=json.loads((pilot/'assets.json').read_text());voices=[v for v in assets['voices'] if v['split']=='test']
 rows=[dict(**r,stem=f'protected-test-{v["engine"]}-{v["name"].lower()}-{i:03}',split='test',engine=v['engine'],voice=v['name'],gender=v['gender'],locale=v['locale'],temperature=.7,supertonic_steps=8,supertonic_speed=1.05) for i,r in enumerate(prompts) for v in voices]
 root=OUT/'confirmation/base';freeze(root/'planned.json',rows);freeze(root/'assets.json',(pilot/'assets.json').read_bytes())
 if not(root/'assets').exists():(root/'assets').symlink_to(pilot/'assets',target_is_directory=True)
 for p in (pilot/'sources').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:freeze(root/'sources'/p.relative_to(pilot/'sources'),p.read_bytes())
 freeze(root/'phone-classes.json',(pilot/'phone-classes.json').read_bytes())
 freeze(OUT/'confirmation/prompts.json',prompts)
 freeze(OUT/'confirmation/audit.json',dict(prior_count=len(prior),prior_manifest_sha256=sources,planned_sha256=sha(root/'planned.json'),source_sha256=sha(__file__),arctic_sha256=sha(SCALE/'text-sources/cmuarctic.data')))
 print('Frozen96 new recordings /24 disjoint texts; prior text count',len(prior),flush=True)
if __name__=='__main__':main()

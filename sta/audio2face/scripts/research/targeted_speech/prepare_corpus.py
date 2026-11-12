"""Mine only training residuals; freeze independent broad/targeted additions and fresh tests."""
from collections import defaultdict,Counter
import re,numpy as np
from common import *
VOWELS=set('aa ae ah ao aw ay eh er ey ih iy ow oy uh uw ax'.split())
def normalize(t):return ' '.join(re.findall('[a-z]+',t.lower()))
def rank(t,salt):return hashlib.sha256((salt+t).encode()).hexdigest()
def main():
 inputs();assert sha(BASELINE)==EXPECTED
 pilot=APP/'.local/compact-coverage-v2/base';lex=json.loads((pilot/'assets/aligner/lexicon.json').read_text())['lexicon']
 audit=json.loads((FEATURE/'confirmation/audit.json').read_text());prior=set();fingerprints={}
 paths=list(audit['prior_manifest_sha256'])+[str(FEATURE/'confirmation/base/planned.json'),str(SCALE/'corpus/planned.json')]
 for name in sorted(set(paths)):
  p=Path(name)
  if not p.exists():raise ValueError('Missing prior transcript inventory: '+name)
  rows=json.loads(p.read_text());prior.update(normalize(r['text']) for r in rows if isinstance(r,dict) and 'text' in r);fingerprints[name]=sha(p)
 def phones(t):return [p.lower() for w in re.findall("[A-Za-z]+(?:'[A-Za-z]+)*",t) for p in lex[w.upper()]]
 def triples(t):
  seq=phones(t);return ['|'.join(seq[i-1:i+2]) for i in range(1,len(seq)-1)]
 candidates=[]
 for line in (SCALE/'text-sources/cmuarctic.data').read_text().splitlines():
  m=re.fullmatch(r'\(\s*(\S+)\s+"(.*)"\s*\)',line)
  if m and normalize(m[2]) not in prior and 5<=len(m[2].split())<=22 and all(w.upper() in lex for w in re.findall("[A-Za-z]+(?:'[A-Za-z]+)*",m[2])):candidates.append(dict(text=m[2],source_id=m[1],text_source='cmu-arctic'))
 candidates.sort(key=lambda r:rank(r['text'],'target-partition-37:'));holdout=candidates[:80];pool=candidates[80:]
 if len(pool)<160:raise ValueError('Not enough independent candidate prompts')
 eligible=Counter(c for r in pool for c in set(triples(r['text'])))
 import mlx.core as mx
 import onnx
 from onnx import numpy_helper
 from mlx_backend import Student
 from features import Dataset
 net=Student({v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer})
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=mx.array(rig['gap_basis'][rig['active']]);rest=mx.array(rig['rest']+rig['gap_offset'])
 data=Dataset(['old-train','new-train']);errors=np.empty((data.length,30),np.float32)
 for start in range(0,data.length,128):
  ix=np.arange(start,min(start+128,data.length));batch=data.batch(ix);pred=mx.clip(net(mx.array(batch['mel']))[:,15:45],0,1)@basis+rest;errors[ix]=100*np.square(np.array(pred)-batch['gaps']).mean(-1)
 np.save(OUT/'train-window-squared-mm.npy',errors)
 stats=defaultdict(lambda:dict(total=0.,frames=0,texts=set(),voices=set()));cursor=0
 rows=sum([[r for r in json.loads((SCALE/'datasets'/n/'manifest.json').read_text()) if r['split']=='train'] for n in ['old-train','new-train']],[])
 for row in rows:
  count=len(range(-8000,row['samples']-4000,8000));batch=data.batch(np.arange(cursor,cursor+count));track=errors[cursor:cursor+count][batch['valid']];cursor+=count;times=np.arange(len(track))/60;ps=row['phones']
  for i in range(1,len(ps)-1):
   key='|'.join(p['phone'] for p in ps[i-1:i+2]);mask=(times>=ps[i]['start'])&(times<ps[i]['end']);n=int(mask.sum())
   if n and eligible[key]>=3:
    r=stats[key];r['total']+=float(track[mask].sum());r['frames']+=n;r['texts'].add(normalize(row['text']));r['voices'].add(row['engine']+':'+row['voice'])
 assert cursor==data.length
 ranked=[]
 for key,r in stats.items():
  if len(r['texts'])>=8 and r['frames']>=100:
   rms=np.sqrt(r['total']/r['frames']);ranked.append(dict(context=key,rms_mm=float(rms),score=float(rms*np.sqrt(len(r['texts'])/(len(r['texts'])+8))),texts=len(r['texts']),frames=r['frames'],voices=len(r['voices']),candidate_texts=eligible[key]))
 ranked.sort(key=lambda r:(-r['score'],r['context']));chosen=[]
 for category,quota in [('bilabial',4),('vowel',8),('other',4)]:
  def cat(r):
   centre=r['context'].split('|')[1];return 'bilabial' if centre in ['m','b','p'] else 'vowel' if centre in VOWELS else 'other'
  chosen.extend([r for r in ranked if cat(r)==category][:quota])
 chosen += [r for r in ranked if r not in chosen][:16-len(chosen)];weights={r['context']:r['score'] for r in chosen}
 def difficulty(r):
  ts=triples(r['text']);return sum(weights.get(t,0) for t in ts)/max(1,len(ts))
 broad=sorted(pool,key=lambda r:rank(r['text'],'broad-37:'))[:72];remaining=[r for r in pool if r not in broad];targeted=[]
 for reference in broad:
  length=len(reference['text'].split());near=[r for r in remaining if abs(len(r['text'].split())-length)<=1]
  if not near:
   delta=min(abs(len(r['text'].split())-length) for r in remaining);near=[r for r in remaining if abs(len(r['text'].split())-length)==delta]
  selected=min(near,key=lambda r:(-difficulty(r),rank(r['text'],'target-37:')));targeted.append(selected);remaining.remove(selected)
 fresh=[]
 for kind,phone_set,n in [('closure-m',{'m'},3),('closure-b',{'b'},3),('closure-p',{'p'},3),('vowels',VOWELS,3),('hard-context',None,4)]:
  available=[r for r in holdout if r not in [x[0] for x in fresh] and (phone_set is None or set(phones(r['text']))&phone_set)]
  for r in sorted(available,key=lambda r:(-difficulty(r),rank(r['text'],'test-hard:')))[:n]:fresh.append((r,kind))
 for r in sorted([r for r in holdout if r not in [x[0] for x in fresh]],key=lambda r:rank(r['text'],'test-broad:'))[:16]:fresh.append((r,'broad-speech'))
 assert len(fresh)==32
 assets=json.loads((pilot/'assets.json').read_text());planned=[];counts=Counter()
 for arm,texts in [('broad',broad),('targeted',targeted),('confirmation',fresh)]:
  for i,item in enumerate(texts):
   r,kind=item if arm=='confirmation' else (item,'training-'+arm)
   voices=[v for v in assets['voices'] if v['split']==('test' if arm=='confirmation' else 'train')]
   if arm!='confirmation':
    names=[('Easton','Alfie')[i%2],('Zoey','Maeve')[(i//2)%2],('M1','M2')[(i//4)%2],('F1','F2')[i%2]];voices=[next(v for v in voices if v['name']==name) for name in names]
   for v in voices:
    key=(arm,v['engine'],v['name']);index=counts[key];counts[key]+=1
    planned.append(dict(**r,source='cmu-arctic-'+('broad' if kind=='broad-speech' else 'hard') if arm=='confirmation' else 'cmu-arctic',kind=kind,arm=arm,split='test' if arm=='confirmation' else 'train',stem=f'target-{arm}-{v["engine"]}-{v["name"].lower()}-{i:03}',engine=v['engine'],voice=v['name'],gender=v['gender'],locale=v['locale'],temperature=.7,supertonic_steps=8,supertonic_speed=1.05,training_gain=['original','half','boost'][index%3] if arm!='confirmation' else None))
 assert len(planned)==704 and len({normalize(r['text']) for r in planned})==176
 for arm in ['broad','targeted']:
  counts_check=Counter((r['engine'],r['voice'],r['training_gain']) for r in planned if r['arm']==arm)
  assert len(counts_check)==24 and set(counts_check.values())=={12},counts_check
 root=OUT/'corpus';freeze(root/'planned.json',planned);freeze(root/'contexts.json',dict(chosen=chosen,ranked=ranked));freeze(root/'audit.json',dict(prior_texts=len(prior),prior_manifests=fingerprints,holdout_texts=holdout,candidate_texts=len(candidates),broad_mean_score=float(np.mean([difficulty(r) for r in broad])),targeted_mean_score=float(np.mean([difficulty(r) for r in targeted])),broad_mean_words=float(np.mean([len(r['text'].split()) for r in broad])),targeted_mean_words=float(np.mean([len(r['text'].split()) for r in targeted]))))
 for engine in ['phonon','supertonic']:
  records=[r for r in planned if r['engine']==engine]
  for shard in range(4):
   view=root/'base-shards'/f'{engine}-{shard:02}';freeze(view/'planned.json',records[shard::4]);freeze(view/'assets.json',(pilot/'assets.json').read_bytes());freeze(view/'phone-classes.json',(pilot/'phone-classes.json').read_bytes())
   if not(view/'assets').exists():(view/'assets').symlink_to(pilot/'assets',target_is_directory=True)
   for p in (pilot/'sources').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:freeze(view/'sources'/p.relative_to(pilot/'sources'),p.read_bytes())
 freeze(root/'protocol.json',dict(source_sha256={'prepare_corpus.py':sha(__file__),'common.py':sha(HERE/'common.py')},planned_sha256=sha(root/'planned.json'),baseline_sha256=EXPECTED,independent_originals=704,confirmation_recordings=128,arctic_sha256=sha(SCALE/'text-sources/cmuarctic.data'),plan_sha256=sha(APP/'docs/plans/compact-targeted-speech.md')))
 print('Frozen704 independent recordings;288 broad/288 targeted/128 confirmation.',flush=True)
 print(json.dumps(json.loads((root/'audit.json').read_text())|{'prior_manifests':'retained','holdout_texts':'80 reserved'},indent=2),flush=True)
if __name__=='__main__':main()

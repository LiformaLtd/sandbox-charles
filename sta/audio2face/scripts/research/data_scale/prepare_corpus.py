"""Freeze real independent speech expansion and disjoint new evaluation prompts."""
from pathlib import Path
import hashlib, json, os, re, shutil
from prompts import authored, DEV, TEST
APP=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('A2F_SCALE_OUTPUT',APP/'.local/compact-scale-v1')).resolve()
WORK=Path(os.environ['A2F_RESEARCH_WORK']).resolve()
PILOT=APP/'.local/compact-coverage-v2/base'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalize(t):return ' '.join(re.findall('[a-z]+',t.lower()))
def freeze(p,v):
    raw=json.dumps(v,indent=2).encode() if isinstance(v,(dict,list)) else v
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_bytes()!=raw:raise ValueError('Frozen artifact changed: '+str(p))
    if not p.exists():p.write_bytes(raw)


def main():
    for arm in ['mixed','fit-mse']:
        if not json.loads((OUT/'memorization'/arm/'result.json').read_text())['memorization_pass']:raise ValueError('Memorization screen did not pass')
    root=OUT/'corpus';root.mkdir(exist_ok=True)
    assets=json.loads((PILOT/'assets.json').read_text());lex=json.loads((PILOT/'assets/aligner/lexicon.json').read_text())['lexicon']
    prior=set()
    paths=[APP/f'.local/{name}/manifest.json' for name in ['compact-multivoice','compact-v4-confirmation']]
    paths += [PILOT/'manifest.json']
    paths += [APP/f'.local/{name}/confirmation/base/planned.json' for name in ['compact-capacity-v1','compact-state-v1','compact-temporal-v1']]
    paths += [WORK/name/'manifest.json' for name in ['arkit-student','arkit-student-v2','arkit-student-v3','arkit-student-v3/extra']]
    for p in paths:prior.update(normalize(r['text']) for r in json.loads(p.read_text()))
    def supported(text):return all(w.upper() in lex for w in re.findall("[A-Za-z]+(?:'[A-Za-z]+)*",text))
    arctic=[]
    for line in (OUT/'text-sources/cmuarctic.data').read_text().splitlines():
        m=re.fullmatch(r'\(\s*(\S+)\s+"(.*)"\s*\)',line)
        if m and supported(m[2]) and normalize(m[2]) not in prior and 5<=len(m[2].split())<=22:
            arctic.append(dict(source='cmu-arctic',source_id=m[1],text=m[2],kind='broad-speech'))
    arctic.sort(key=lambda r:hashlib.sha256(('scale-37:'+r['text']).encode()).hexdigest())
    if len(arctic)<608:raise ValueError('Not enough unseen supported Arctic text')
    selections={
      'train':arctic[:512]+[dict(source='authored',source_id=key,kind=k,text=t) for k,t,key in authored()],
      'dev':arctic[512:544]+[dict(source='authored',source_id='dev-'+str(i),kind=k,text=t) for i,(k,t) in enumerate(DEV)],
      'test':arctic[544:608]+[dict(source='authored',source_id='test-'+str(i),kind=k,text=t) for i,(k,t) in enumerate(TEST)],
    }
    seen=set(prior)
    for split,rows in selections.items():
        for r in rows:
            key=normalize(r['text'])
            if key in seen:raise ValueError('Repeated/historical prompt: '+r['text'])
            if not supported(r['text']):raise ValueError('Unknown words: '+str([w for w in re.findall("[A-Za-z]+(?:'[A-Za-z]+)*",r['text']) if w.upper() not in lex]))
            seen.add(key)
    counts={};planned=[]
    for split in ['dev','test','train']:
        voices=[v for v in assets['voices'] if v['split']==split]
        for i,text in enumerate(selections[split]):
            selected=voices
            if split=='train':
                selected=[next(v for v in voices if v['name']==n) for n in [('Easton','Alfie')[i%2],('Zoey','Maeve')[(i//2)%2],('M1','M2')[(i//4)%2],('F1','F2')[(i//8)%2]]]
            for voice in selected:
                key=voice['engine']+':'+voice['name'];index=counts.get(key,0);counts[key]=index+1
                planned.append(dict(**text,stem=f'scale-{split}-{voice["engine"]}-{voice["name"].lower()}-{i:04}',split=split,
                    engine=voice['engine'],voice=voice['name'],gender=voice['gender'],locale=voice['locale'],temperature=.7,
                    supertonic_steps=8,supertonic_speed=1.05,training_gain=['original','half','boost'][index%3] if split=='train' else None))
    if len([r for r in planned if r['split']=='train'])!=3072 or len(planned)!=3616:raise ValueError('Unexpected corpus size')
    freeze(root/'texts.json',selections);freeze(root/'planned.json',planned)
    for engine,shards in [('phonon',4),('supertonic',8)]:
        rows=[r for r in planned if r['engine']==engine]
        for shard in range(shards):
            view=root/'base-shards'/f'{engine}-{shard:02}';view.mkdir(parents=True,exist_ok=True)
            freeze(view/'planned.json',rows[shard::shards]);freeze(view/'assets.json',(PILOT/'assets.json').read_bytes())
            if not (view/'assets').exists():(view/'assets').symlink_to(PILOT/'assets',target_is_directory=True)
            for p in (PILOT/'sources').rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:freeze(view/'sources'/p.relative_to(PILOT/'sources'),p.read_bytes())
            freeze(view/'phone-classes.json',(PILOT/'phone-classes.json').read_bytes())
    protocol=dict(text_source_sha256=sha(OUT/'text-sources/cmuarctic.data'),text_source_commit='c51cb971036a127359b43c621f26ca5081eab13e',
        source_url='https://raw.githubusercontent.com/rhasspy/dataset-voice-kathleen/c51cb971036a127359b43c621f26ca5081eab13e/cmuarctic.data',
        original_source='https://festvox.org/cmu_arctic/cmuarctic.data',report='https://festvox.org/cmu_arctic/cmu_arctic_report.pdf',
        planned_sha256=sha(root/'planned.json'),texts_sha256=sha(root/'texts.json'),assets_sha256=sha(PILOT/'assets.json'),
        source_sha256={p.name:sha(p) for p in [HERE/'prepare_corpus.py',HERE/'prompts.py']},counts=counts,
        primary_new_originals=3072,development_originals=192,confirmation_originals=352,
        augmentation='One teacher-labelled gain level per new training original, balanced per voice; all 3 dev and 4 test levels',
        training_expected_per_voice=384,expected_training_per_voice_per_gain=128,prior_normalized_text_count=len(prior))
    freeze(root/'protocol.json',protocol)
    for name in protocol['source_sha256']:freeze(root/'frozen'/name,(HERE/name).read_bytes())
    print('Frozen',len(planned),'new independent recordings; 3072 train, 192 dev, 352 confirmation.')


if __name__=='__main__':main()

from pathlib import Path
import os
import importlib.util,json,re,math,collections
APP=Path(__file__).resolve().parents[3]
HERE=Path(os.environ.get('A2F_ABLATION_OUTPUT',APP/'.local/compact-ablation')).resolve()
HERE.mkdir(parents=True,exist_ok=True)
WORK=Path(os.environ.get('A2F_RESEARCH_WORK',APP/'.local/research-corpora')).resolve()
V3=WORK/'arkit-student-v3'
import evaluation_support as ev
cache=HERE/'rank-v3';cache.mkdir(exist_ok=True)
session=ev.session(V3/'james-arkit-student-v3.onnx')
records=ev.collect(V3,session,cache,'train')+ev.collect(V3/'extra',session,cache,'train')
ranking=ev.rank_pairs(records);(HERE/'training-pair-errors.json').write_text(json.dumps(ranking,indent=2))
errors={r['pair']:r for r in ranking if r['count']>=8}
norm=lambda s:' '.join(re.findall('[a-z0-9]+',s.lower()))
roots=[WORK/'arkit-student',WORK/'arkit-student-v2',V3,V3/'extra']
known={norm(r['text']) for root in roots for r in json.loads((root/'manifest.json').read_text())}
SOURCE=Path(os.environ.get('A2F_DICT_ROOT',APP.parent/'audio2face-dict')).resolve()
lex={k.lower():v for k,v in json.loads((SOURCE/'static/models/lexicon.json').read_text())['lexicon'].items()}
freq={r['pair']:r['occurrences'] for r in json.loads((SOURCE/'.local/subtitle-analysis/pair-frequencies.json').read_text())}
candidates=[]
for row in json.loads((SOURCE/'.local/subtitle-analysis/standalone.json').read_text()):
    if norm(row['text']) in known:continue
    words=re.findall("[a-z]+(?:'[a-z]+)*",row['text'].lower())
    if not words or not all(w in lex for w in words) or max(map(len,words))>13:continue
    phones=[p for word in words for p in lex[word]]
    pairs=[' '.join(v) for v in zip(phones,phones[1:])]
    candidates.append(dict(text=row['text'],pairs=pairs))
seen=collections.Counter();chosen=[]
while candidates and len(chosen)<64:
    def score(row):
        return sum((errors[p]['lip_gap_rmse']**2+.2*errors[p]['contact_error'])*math.log1p(freq.get(p,0))/(1+seen[p]) for p in set(row['pairs']) if p in errors)/math.sqrt(len(row['pairs']))
    row=max(candidates,key=score);gain=score(row)
    if gain<=0:break
    candidates.remove(row);seen.update(set(row['pairs']))
    chosen.append(dict(stem=f'train-ablation-extra-{len(chosen):03d}',split='train',text=row['text'],source='extra',selection_score=gain))
assert len(chosen)>=48,len(chosen)
assert not {norm(r['text']) for r in chosen}&known
(HERE/'extra-prompts.json').write_text(json.dumps(chosen,indent=2))
print('Selected',len(chosen),'new training prompts; no exact normalized overlap with any existing split.',flush=True)

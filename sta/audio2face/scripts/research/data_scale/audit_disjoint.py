"""Check frozen new texts against every retained prior manifest before training."""
from pathlib import Path
import json,re
from prepare_corpus import APP,OUT,WORK,sha,freeze


def main():
    normal=Path('/Users/charleswiles/Source/liforma/sandboxes/sandbox-charles/sta/audio2face/.local')
    paths=list(normal.rglob('manifest.json'))+list(normal.rglob('planned.json'))
    paths += [WORK/p/'manifest.json' for p in ['arkit-student','arkit-student-v2','arkit-student-v3','arkit-student-v3/extra']]
    paths += [APP/'.local/compact-temporal-v1/confirmation/base/planned.json']
    norm=lambda t:' '.join(re.findall('[a-z]+',t.lower()))
    previous=set();sources={}
    for p in sorted(set(paths)):
        try:rows=json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):continue
        if not isinstance(rows,list):continue
        texts=[r['text'] for r in rows if isinstance(r,dict) and isinstance(r.get('text'),str)]
        if texts:previous.update(map(norm,texts));sources[str(p)]=sha(p)
    texts=json.loads((OUT/'corpus/texts.json').read_text())
    overlap=[dict(split=split,text=r['text']) for split,rows in texts.items() for r in rows if norm(r['text']) in previous]
    result=dict(prior_normalized_texts=len(previous),source_sha256=sources,overlap=overlap)
    freeze(OUT/'disjoint-audit.json',result)
    if overlap:raise ValueError('Historical overlap: '+str(overlap))
    print('No overlap with',len(previous),'retained prior normalized texts across',len(sources),'manifests.')


if __name__=='__main__':main()

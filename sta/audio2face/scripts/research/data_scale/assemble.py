"""Assemble matched data arms and one shared development set after labelling."""
from pathlib import Path
import json
import numpy as np
from prepare_corpus import APP,OUT,WORK,HERE,freeze,sha


def main():
    corpus=OUT/'corpus'
    if not (corpus/'complete.json').exists():raise ValueError('Corpus pipeline incomplete')
    old=APP/'.local/compact-coverage-v2/coverage';prior=json.loads((old/'manifest.json').read_text())
    collected=[];recipes=[];labels={}
    for root in sorted((corpus/'label-shards').iterdir()):
        rows=json.loads((root/'manifest.json').read_text());recipe=json.loads((root/'label-recipe.json').read_text());recipes.append(recipe)
        for row in rows:
            p=root/'labels'/(row['stem']+'.npz')
            if sha(row['audio'])!=row['sha256_audio']:raise ValueError('Changed waveform')
            with np.load(p) as z:
                frames=(row['samples']*60+15999)//16000
                if z['weights'].shape!=(frames,52) or z['gaps'].shape!=(frames,9) or str(z['audio_sha256'])!=row['sha256_audio']:raise ValueError('Invalid label track')
            collected.append(row);labels[row['stem']]=p
    if any(r!=recipes[0] for r in recipes):raise ValueError('Mixed new teacher recipes')
    for row in prior:labels[row['stem']]=old/'labels'/(row['stem']+'.npz')
    common_dev=[r for r in prior if r['split']=='dev']+[r for r in collected if r['split']=='dev']
    sets={
      'old-train':[r for r in prior if r['split']=='train'],
      'new-train':[r for r in collected if r['split']=='train'],
      'common-dev':common_dev,
      'confirmation':[r for r in collected if r['split']=='test'],
    }
    if [len(sets[n]) for n in sets]!=[768,3072,864,1408]:raise ValueError('Unexpected split sizes')
    for name,rows in sets.items():
        root=OUT/'datasets'/name;(root/'labels').mkdir(parents=True,exist_ok=True)
        freeze(root/'manifest.json',rows)
        for row in rows:
            p=root/'labels'/(row['stem']+'.npz')
            if not p.exists():p.symlink_to(labels[row['stem']])
    freeze(OUT/'datasets/teacher-recipe-new.json',recipes[0])
    freeze(OUT/'assembly-lock.json',dict(source_sha256=sha(__file__),manifests={n:sha(OUT/'datasets'/n/'manifest.json') for n in sets},
        old_label_recipe_sha256=sha(old/'label-recipe.json') if (old/'label-recipe.json').exists() else None,new_teacher_recipe_sha256=sha(OUT/'datasets/teacher-recipe-new.json')))
    print({n:len(v) for n,v in sets.items()})


if __name__=='__main__':main()

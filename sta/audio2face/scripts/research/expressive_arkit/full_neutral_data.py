"""Absolute teacher face targets, regional brows and raw geometric lip gaps."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
from common import *
from fit import bounded_solve,mm

FULL=Path(os.environ.get('A2F_NEUTRAL_OUTPUT',APP/'.local/expressive-full-neutral-v1')).resolve()


def labels():
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import animate,load_backend,MODEL_SHA256
    with np.load(OUT/'calibration.npz') as z:shape=z['shape'];neutral=z['neutral']
    with np.load(ASSETS/'model_data_James.npz') as z:model_neutral=z['neutral_skin'];tongue_neutral=z['neutral_tongue']
    from fit import PAIRS
    upper,lower=PAIRS.T
    support=np.max(np.linalg.norm(shape[41:46],axis=2),axis=0);region=np.argsort(support)[-512:]
    basis=shape[41:46][:,region].reshape(5,-1)
    gram=mm(basis,basis.T);gram+=np.diag(np.diag(gram)*.0001)
    others=shape[:,region].copy();others[41:46]=0
    tip=np.argsort(tongue_neutral[:,2])[-56:]
    rows=json.loads((OUT/'manifest.json').read_text())
    recipe=dict(teacher_sha256=MODEL_SHA256,prior_labels_sha256=sha(OUT/'label-recipe.json'),
        source_sha256=sha(__file__),identity='James',noise_seed=0,emotions=['neutral',*EMOTIONS],
        raw_lip_gap_pairs=PAIRS.tolist(),brow_region=region.tolist(),
        brow_fit='5-channel bounded regional solve after subtracting fitted other-face motion',
        tongue='Prior stretch proxy gated by actual raw teacher tip protrusion beyond moving inner-lip depth; 1mm smooth transition.',
        neutral='Absolute teacher motion; no v4 neutral target or gate')
    freeze(FULL/'label-recipe.json',recipe);recipe_sha=sha(FULL/'label-recipe.json')
    backend=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    (FULL/'labels').mkdir(exist_ok=True);started=time.monotonic()
    for i,row in enumerate(rows):
        if sha(row['audio'])!=row['sha256_audio']:raise ValueError('Changed source audio')
        pcm=np.fromfile(row['audio'],'<f4')
        for ei,emotion_name in enumerate(['neutral',*EMOTIONS]):
            case=row['stem']+'--'+emotion_name;dest=FULL/'labels'/f'{case}.npz'
            if dest.exists():
                with np.load(dest) as z:
                    if str(z['recipe_sha256'])!=recipe_sha:raise ValueError('Changed label recipe')
                continue
            with np.load(OUT/'labels'/f'{case}.npz') as z:prior=z['weights'].copy()
            control=np.zeros((1,30,10),np.float32)
            if ei:control[:,:,ei-1]=1
            def infer(inputs):return backend({**inputs,'emotion':control})
            weights=[];gaps=[];protrusion=[];cursor=0
            for indices,geometry in animate(pcm,infer,identity='James',seed=0):
                w=prior[cursor:cursor+len(indices)].copy();cursor+=len(indices)
                skin=geometry[:,:72006].reshape(-1,24002,3)+model_neutral
                desired=skin[:,region]-neutral[region]-np.einsum('fc,cvk->fvk',w,others,optimize=False)
                w[:,41:46]=bounded_solve(mm(desired.reshape(len(w),-1),basis.T),gram)
                tongue=geometry[:,72006:88812].reshape(-1,5602,3)+tongue_neutral
                tip_z=np.quantile(tongue[:,tip,2],.95,axis=1)
                lip_z=np.mean((skin[:,upper[3:6],2]+skin[:,lower[3:6],2])*.5,axis=1)
                exposure=tip_z-lip_z
                gate=np.clip(exposure/.1,0,1);gate=gate*gate*(3-2*gate)
                w[:,51]*=gate
                weights.append(w);gaps.append(skin[:,upper,1]-skin[:,lower,1]);protrusion.append(exposure)
            if cursor!=len(prior):raise ValueError('Teacher timestamp/count mismatch')
            np.savez_compressed(dest,weights=np.concatenate(weights),gaps=np.concatenate(gaps),
                tongue_exposure=np.concatenate(protrusion),recipe_sha256=recipe_sha,audio_sha256=row['sha256_audio'])
        print('Full neutral labels',i+1,'/',len(rows),'seconds',round(time.monotonic()-started),flush=True)


def features():
    for split in ['train','dev','test']:
        with np.load(OUT/'features'/f'{split}.npz') as z:base={k:z[k].copy() for k in z.files}
        target=np.zeros((len(base['cases']),30,52),np.float32);gaps=np.zeros((len(target),30,9),np.float32)
        cache={};cursor={}
        for i,name in enumerate(base['cases']):
            case=str(name)
            if case not in cache:
                with np.load(FULL/'labels'/f'{case}.npz') as z:cache[case]=(z['weights'].copy(),z['gaps'].copy())
                cursor[case]=0
            count=int(base['valid'][i].sum());start=cursor[case]
            target[i,base['valid'][i]]=cache[case][0][start:start+count]
            gaps[i,base['valid'][i]]=cache[case][1][start:start+count]
            cursor[case]+=count
        if any(cursor[k]!=len(cache[k][0]) for k in cache):raise ValueError('Feature alignment failed')
        np.savez_compressed(FULL/f'{split}.npz',hidden=base['hidden'],baseline=base['baseline'],target=target,
            gaps=gaps,emotion=base['emotion'],valid=base['valid'],voices=base['voices'],cases=base['cases'])
        print(split,len(target),'absolute full-face windows',flush=True)


def replay():
    import torch
    from prepare import baseline
    torch.set_num_threads(4)
    _,student,_=baseline();student.eval()
    root=SOURCE_APP/'.local/compact-scale-v1/features'
    for name in ['replay-train','replay-dev','common-dev']:
        source=root/name
        mel=np.load(source/'mel.npy',mmap_mode='r');length=len(mel)
        # Development remains complete. Replay training is already a compact
        # historical corpus and is not selected using the three regression clips.
        hidden=[];coefficients=[]
        with torch.inference_mode():
            for start in range(0,length,128):
                x=torch.from_numpy(mel[start:start+128].copy());h=student.blocks(torch.nn.functional.gelu(student.encoder(x),approximate='tanh'))
                aligned=h.index_select(2,student.left)+student.fraction*(h.index_select(2,student.left+1)-h.index_select(2,student.left))
                hidden.append(aligned.transpose(1,2).numpy());coefficients.append(student.coefficients(x).numpy())
        np.savez_compressed(FULL/f'{name}.npz',hidden=np.concatenate(hidden),baseline=np.concatenate(coefficients),
            gaps=np.load(source/'gaps.npy'),valid=np.load(source/'valid.npy'),voices=np.load(source/'voice.npy'))
        freeze(FULL/(name+'-source.json'),{p.name:sha(p) for p in source.glob('*.npy')})
        print(name,length,'mouth replay/validation windows',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['labels','features','replay']);args=parser.parse_args()
    FULL.mkdir(parents=True,exist_ok=True)
    {'labels':labels,'features':features,'replay':replay}[args.stage]()

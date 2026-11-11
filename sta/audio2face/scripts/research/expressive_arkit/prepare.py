"""Re-run full NVIDIA output under each emotion; retain all derived components."""
import argparse
import collections
import importlib.util
import os
import sys
import time
os.environ['MLX_ENABLE_TF32'] = '0'
import numpy as np
import onnx
from onnx import numpy_helper
from common import *
from fit import FullFaceFit


def model_source():
    source = APP / 'scripts/research/compact_ablation/model.py'
    spec = importlib.util.spec_from_file_location('compact_base', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline():
    import torch
    if sha(BASELINE) != EXPECTED_BASELINE:
        raise ValueError('Unexpected frozen speech model')
    values = {v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer}
    active = np.argmax(values['student.channel_map'],1)
    base = model_source()
    student = base.Student(96,values['student.mean'],values['student.scale'],active)
    student.load_state_dict({k:torch.from_numpy(values['student.'+k]) for k in student.state_dict()})
    student.eval()
    return base,student,active


def manifest():
    source = SOURCE_APP / '.local/compact-multivoice/manifest.json'
    groups = collections.defaultdict(list)
    for row in json.loads(source.read_text()):
        groups[(row['split'],row['engine'],row['voice'])].append(row)
    selected = []
    for (split,engine,voice), rows in sorted(groups.items()):
        count = 4 if split == 'train' else 2
        for row in rows[:count]:
            audio = Path(row['audio'])
            if sha(audio) != row['sha256_audio']:
                raise ValueError('Changed source audio: '+str(audio))
            selected.append({**row,'emotion_conditions':['neutral',*EMOTIONS]})
    texts = {s:{r['text'] for r in selected if r['split']==s} for s in ['train','dev','test']}
    if any(texts[a] & texts[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):
        raise ValueError('Prompt leakage across adapter splits')
    freeze(OUT/'manifest.json',selected)
    freeze(OUT/'label-recipe.json',dict(teacher_root=str(TEACHER),assets=str(ASSETS),
        teacher_sha256=sha(TEACHER/'models_cache/audio2face-v3/network.onnx'),
        assets_sha256={p.name:sha(p) for p in ASSETS.glob('*James*')},
        scene_sha256=sha(WORK/'arkit-student/james_arkit_v3.ma'),
        fit_sha256=sha(Path(__file__).with_name('fit.py')), source_sha256=sha(__file__),
        baseline_sha256=sha(BASELINE),manifest_sha256=sha(OUT/'manifest.json'),
        emotions=EMOTIONS,emotion_intensity=1.,identity='James',noise_seed=0,mlx_tf32=False,
        gaze_calibration='25 degrees in each signed rig-space axis = coefficient 1; experimental endpoint',
        tongue_calibration='canonical tongueStretch after authoring mouth/jaw subtraction; experimental tongueOut proxy',
        head_motion='not present in released NVIDIA export; deferred; no DAWN labels',
        caveat='Known speech recordings; held out from this adapter training, not fresh independent speech confirmation.'))
    return selected


def label():
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import load_backend, animate
    rows = manifest()
    fitter = FullFaceFit(ASSETS,WORK/'arkit-student/james_arkit_v3.ma',APP)
    fitter.save_calibration(OUT/'calibration.npz')
    freeze(OUT/'channels.json',fitter.names)
    original = load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    recipe_sha = sha(OUT/'label-recipe.json')
    (OUT/'labels').mkdir(exist_ok=True)
    started = time.monotonic()
    for i,row in enumerate(rows):
        pcm = np.fromfile(row['audio'],'<f4')
        for index,name in enumerate(['neutral',*EMOTIONS]):
            target = OUT/'labels'/f'{row["stem"]}--{name}.npz'
            if target.exists():
                with np.load(target) as z:
                    if str(z['recipe_sha256'])!=recipe_sha or str(z['audio_sha256'])!=row['sha256_audio']:
                        raise ValueError('Changed label provenance: '+str(target))
                continue
            emotion = np.zeros((1,30,10),np.float32)
            if index:emotion[:,:,index-1] = 1
            def infer(inputs):return original({**inputs,'emotion':emotion})
            weights, components = [],collections.defaultdict(list)
            for indices,geometry in animate(pcm,infer,identity='James',seed=0):
                w,extra = fitter(geometry)
                weights.append(w)
                for key,value in extra.items():components[key].append(value)
            w=np.concatenate(weights)
            if w.shape!=((len(pcm)*60+15999)//16000,52) or not np.isfinite(w).all():
                raise ValueError('Incomplete coefficient labels')
            temporary = target.with_suffix('.part.npz')
            np.savez_compressed(temporary,weights=w,**{k:np.concatenate(v) for k,v in components.items()},
                                recipe_sha256=recipe_sha,audio_sha256=row['sha256_audio'])
            temporary.replace(target)
        print('Labeled',i+1,'/',len(rows),row['stem'],'seconds',round(time.monotonic()-started,1),flush=True)


def features():
    import torch
    torch.set_num_threads(4)
    base,net,active=baseline()
    rows=json.loads((OUT/'manifest.json').read_text())
    (OUT/'features').mkdir(exist_ok=True)
    for split in ['train','dev','test']:
        xs,bs,ys,es,ms,vs,cs=[],[],[],[],[],[],[]
        for row in rows:
            if row['split']!=split:continue
            pcm=np.fromfile(row['audio'],'<f4')
            labels=[]
            for name in ['neutral',*EMOTIONS]:
                with np.load(OUT/'labels'/f'{row["stem"]}--{name}.npz') as z:labels.append(z['weights'].copy())
            cursor=0
            for start in range(-8000,len(pcm)-4000,8000):
                audio=np.zeros(16000,np.float32)
                lo,hi=max(start,0),min(start+16000,len(pcm))
                audio[lo-start:hi-start]=pcm[lo:hi]
                windows=np.lib.stride_tricks.sliding_window_view(np.pad(audio,(200,200)),400)[::160]
                fft=np.fft.rfft(windows*base.HANN,n=512,axis=-1)
                power=(fft.real**2+fft.imag**2).astype(np.float32)
                mel=np.log1p(np.einsum('tf,mf->mt',power,base.MEL,optimize=False)*10)
                with torch.inference_mode():
                    m=torch.from_numpy(mel[None])
                    hidden=net.blocks(torch.nn.functional.gelu(net.encoder(m),approximate='tanh'))
                    aligned=hidden.index_select(2,net.left)+net.fraction*(hidden.index_select(2,net.left+1)-hidden.index_select(2,net.left))
                    b=net.coefficients(m)[0].numpy()
                    hidden=aligned[0].numpy().T
                positions=start+np.arange(15,45)*16000//60
                mask=(positions>=0)&(positions<len(pcm));count=int(mask.sum())
                for ei,name in enumerate(['neutral',*EMOTIONS]):
                    target=np.zeros((30,52),np.float32)
                    # Paired expression differences retain the released articulation
                    # at neutral, rather than re-learning the existing mouth path.
                    reference=labels[0][cursor:cursor+count]
                    desired=labels[ei][cursor:cursor+count].copy()
                    desired[:,active]=np.clip(b[15:45][mask][:,active] + desired[:,active]-reference[:,active],0,1)
                    target[mask]=desired
                    emotion=np.zeros(10,np.float32)
                    if ei:emotion[ei-1]=1
                    xs.append(hidden);bs.append(b);ys.append(target);es.append(emotion);ms.append(mask)
                    vs.append(row['engine']+':'+row['voice']);cs.append(row['stem']+'--'+name)
                cursor+=count
            if cursor!=len(labels[0]):raise ValueError('Feature/label timestamp mismatch')
        np.savez_compressed(OUT/'features'/f'{split}.npz',hidden=np.stack(xs),baseline=np.stack(bs),
            target=np.stack(ys),emotion=np.stack(es),valid=np.stack(ms),voices=np.array(vs),cases=np.array(cs),active=active)
        print(split,'windows',len(xs),'voices',len(set(vs)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['labels','features'])
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    (label if args.stage=='labels' else features)()

"""Replay raw NVIDIA teacher and compare brow geometry on identical neutral audio."""
import argparse
import json
import os
import sys
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from infer_full import sequence


def main(sample_id,emotion='neutral'):
    public=APP/'public/assets/expressive';manifest=json.loads((public/'review.json').read_text())
    sample=next(s for s in manifest['samples'] if s['id']==sample_id)
    audio_path=APP/'public'/sample['pcm'].lstrip('/')
    if sha(audio_path)!=sample['sha256']:raise ValueError('Changed audio')
    pcm=np.fromfile(audio_path,'<f4');candidate=APP/'public'/manifest['candidate']['url'].lstrip('/')
    if sha(candidate)!=manifest['candidate']['sha256']:raise ValueError('Changed candidate')
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import load_backend,render,MODEL_SHA256
    backend=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    control=np.zeros((1,30,10),np.float32)
    if emotion!='neutral':control[:,:,EMOTIONS.index(emotion)]=1
    def infer(inputs):return backend({**inputs,'emotion':control})
    result=render(pcm,infer,identity='James',seed=0)
    indices=result['sample_indices'];count=len(indices)
    if not np.array_equal(indices,np.arange(count)*16000//60):raise ValueError('Teacher timeline differs from audio frame grid')
    geometry=np.concatenate([result['skin'].reshape(count,-1),result['tongue'].reshape(count,-1),result['jaw'].reshape(count,-1),result['eyes']],axis=1).astype('<f4')
    if geometry.shape!=(int(np.ceil(len(pcm)*60/16000)),88831) or not np.isfinite(geometry).all():raise ValueError('Invalid teacher track')
    rig=json.loads((public/'rig.json').read_text());binary=(public/'rig.bin').read_bytes();mesh=rig['meshes']['skin']
    rig_neutral=np.frombuffer(binary,dtype='<f4',count=mesh['vertexCount']*3,offset=mesh['positionOffset']).reshape(-1,3)
    with np.load(ASSETS/'model_data_James.npz') as z:teacher_neutral=z['neutral_skin']
    if not np.array_equal(rig_neutral,teacher_neutral):raise ValueError('Raw teacher and browser rest mesh differ')
    model=sequence(candidate,pcm,emotion,1.,'James')['blendshapes']
    with np.load(OUT/'calibration.npz') as z:shape=z['shape'];neutral=z['neutral']
    fitted_path=APP/'.local/expressive-full-neutral-v1/labels'/f'{sample_id}--{emotion}.npz'
    if fitted_path.exists():
        with np.load(fitted_path) as z:fitted=z['weights']
    else:
        from fit import FullFaceFit
        fitter=FullFaceFit(ASSETS,WORK/'arkit-student/james_arkit_v3.ma',APP)
        fitted,_=fitter(geometry)
    if len(model)!=count or len(fitted)!=count:raise ValueError('Teacher/student frame counts differ')
    landmarks=np.argmax(np.linalg.norm(shape[41:46],axis=2),axis=1)
    direction=shape[np.arange(41,46),landmarks];direction/=np.linalg.norm(direction,axis=1)[:,None]
    surface=dict(teacher_raw=result['skin']+teacher_neutral-neutral,
        teacher_fitted=np.einsum('fc,cvk->fvk',fitted,shape,optimize=False),
        candidate=np.einsum('fc,cvk->fvk',model,shape,optimize=False))
    signal={k:np.einsum('fpk,pk->fp',v[:,landmarks],direction,optimize=False)*10 for k,v in surface.items()}
    swing={k:np.percentile(v[15:-15],95,axis=0)-np.percentile(v[15:-15],5,axis=0) for k,v in signal.items()}
    support=np.max(np.linalg.norm(shape[41:46],axis=2),axis=0);region=np.argsort(support)[-512:]
    rms={k:float(np.sqrt(np.mean(((v-surface['teacher_raw'])[:,region]*10)**2))) for k,v in surface.items() if k!='teacher_raw'}
    out=APP/'.local'/('teacher-review-'+sample_id+('' if emotion=='neutral' else '--'+emotion));out.mkdir(exist_ok=True)
    track=out/'teacher.f32';geometry.tofile(track);name='teacher-'+sample_id+'-'+sha(track)[:12]+'.f32'
    import shutil
    shutil.copyfile(track,public/name)
    reference=dict(sample=sample_id,url='/assets/expressive/'+name,sha256=sha(track),frames=count,
        channels=88831,fps=60,audioSha256=sample['sha256'],teacherModelSha256=MODEL_SHA256,
        emotion=emotion,identity='James',seed=0,format='Raw skin, tongue and jaw deltas, then eye rotations; no ARKit fitting')
    references={r['sample']+'--'+r['emotion']:r for r in manifest.get('teacherReviews',{}).values()}
    if manifest.get('teacherReview'):
        old=manifest['teacherReview'];references[old['sample']+'--'+old['emotion']]=old
    references[sample_id+'--'+emotion]=reference
    manifest['teacherReviews']=references
    manifest['teacherReview']=reference
    temp=public/'review.json.tmp';temp.write_text(json.dumps(manifest,indent=2));temp.replace(public/'review.json')
    report=dict(reference=reference,candidate_sha256=sha(candidate),text=sample['text'],
        audio_sha256=sample['sha256'],controls=emotion+' (zero controls for neutral, otherwise full intensity); James identity; teacher seed 0',
        landmark_vertex_indices=landmarks.tolist(),robust_brow_swing_mm={k:v.tolist() for k,v in swing.items()},
        raw_teacher_brow_region_geometry_rms_mm=rms,
        limitations=['Known training recording selected to demonstrate motion, not independent validation.',
          'Raw teacher has a denser facial motion space than the ARKit rig. Teacher fitted track shows retargeting loss separately.',
          'Displayed face differences include all teacher facial motion; no artificial brow amplification.'])
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'brow-traces.npz',**signal)
    fig,axes=plt.subplots(5,1,figsize=(10,10),sharex=True)
    names=['browDownLeft landmark','browDownRight landmark','browInnerUp landmark','browOuterUpLeft landmark','browOuterUpRight landmark']
    for i,ax in enumerate(axes):
        for key,label in [('teacher_raw','NVIDIA raw'),('teacher_fitted','Teacher fitted to ARKit'),('candidate','Our candidate')]:
            values=signal[key][:,i];ax.plot(np.arange(count)/60,values-values.mean(),label=label)
        ax.set_ylabel('mm');ax.set_title(names[i],fontsize=10);ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel('Audio time (seconds)')
    fig.suptitle('Same recording · '+emotion+' · mean-centered brow movement')
    fig.tight_layout();fig.savefig(out/'brow-comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sample',default='train-phonon-maeve-001');parser.add_argument('--emotion',choices=['neutral',*EMOTIONS],default='neutral');args=parser.parse_args();main(args.sample,args.emotion)

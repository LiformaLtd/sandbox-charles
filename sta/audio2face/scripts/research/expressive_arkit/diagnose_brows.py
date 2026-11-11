"""Locate eyebrow vibration in raw teacher, fitting, hybrid targets and student."""
import argparse
import json
import os
import sys
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
from scipy.signal import periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from gaze import NEW
from fit import FullFaceFit
from infer_full import read_audio, sequence
from infer import sequence as baseline_sequence


def linear_detrend(values):
    x=np.asarray(values,np.float64)
    time=np.arange(len(x),dtype=np.float64);time-=time.mean()
    centered=x-x.mean(0)
    slope=np.einsum('i,ij->j',time,centered,optimize=False)/np.einsum('i,i->',time,time,optimize=False)
    return centered-time[:,None]*slope


def temporal_metrics(values):
    """RMS over coordinates; periodogram integrates Hz into position variance."""
    x=np.asarray(values,np.float64)[15:-15].reshape(len(values)-30,-1)
    f,p=periodogram(linear_detrend(x),fs=60,window='hann',detrend=False,scaling='density',axis=0)
    df=f[1]-f[0]
    def band(lo,hi):return float(np.sqrt(np.mean(np.sum(p[(f>=lo)&(f<hi)],axis=0)*df)))
    delta=np.diff(x,axis=0)
    absolute_frames=np.arange(16,len(values)-15)
    join=(absolute_frames-15)%30==0
    return dict(slow_0_4_hz_rms=band(0,4),fast_4_15_hz_rms=band(4,15),fast_15_30_hz_rms=band(15,30.001),
                frame_step_rms=float(np.sqrt(np.mean(delta**2))),
                window_join_step_rms=float(np.sqrt(np.mean(delta[join]**2))),
                other_step_rms=float(np.sqrt(np.mean(delta[~join]**2))))


def main(sample_id='appointment'):
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
    sample=next(s for s in manifest['samples'] if s['id']==sample_id)
    out=NEW/('brow-diagnostic-'+sample_id);out.mkdir(exist_ok=True)
    if sample_id=='appointment':
        source=Path('/Users/charleswiles/Source/liforma/sandboxes/sandbox-charles/sta/audio2face-2d/static/assets/speech/appointment-supertonic-m1.wav')
        audio=read_audio(source)
    else:
        source=APP/'public'/sample['pcm'].lstrip('/')
        audio=np.fromfile(source,'<f4')
        if sha(source)!=sample['sha256']:raise ValueError('Changed source recording')
    fitter=FullFaceFit(ASSETS,WORK/'arkit-student/james_arkit_v3.ma',APP)
    brows=np.arange(41,46)
    support=np.max(np.linalg.norm(fitter.shape[brows],axis=2),axis=0)
    region=np.argsort(support)[-512:]
    landmarks=np.argmax(np.linalg.norm(fitter.shape[brows],axis=2),axis=1)
    direction=fitter.shape[brows,landmarks]
    direction=direction/np.linalg.norm(direction,axis=1)[:,None]
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import load_backend,animate,MODEL_SHA256
    backend=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    coefficients={};surface={};points={}
    for name,seed in [('neutral',0),('anger',0),('anger',1)]:
        label='teacher_'+name+('_seed1' if seed else '')
        control=np.zeros((1,30,10),np.float32)
        if name!='neutral':control[:,:,EMOTIONS.index(name)]=1
        def infer(inputs):return backend({**inputs,'emotion':control})
        ws,rs,ps=[],[],[]
        for indices,geometry in animate(audio,infer,identity='James',seed=seed):
            w,_=fitter(geometry);ws.append(w)
            skin=geometry[:,:72006].reshape(-1,24002,3)
            rs.append((skin[:,region]+fitter.model_neutral[region]-fitter.neutral[region])*10)
            ps.append(np.einsum('fpk,pk->fp',skin[:,landmarks]+fitter.model_neutral[landmarks]-fitter.neutral[landmarks],direction,optimize=False)*10)
        coefficients[label]=np.concatenate(ws);surface[label+'_raw']=np.concatenate(rs);points[label+'_raw']=np.concatenate(ps)
        print('measured',label,flush=True)
    coefficients['current_v4']=baseline_sequence(BASELINE,audio)
    coefficients['candidate']=sequence(NEW/'fullface.onnx',audio,'anger',1.,'Claire')['blendshapes']
    with np.load(WORK/'arkit-student/rig-metrics.npz') as z:active=z['active']
    hybrid=coefficients['teacher_anger'].copy()
    hybrid[:,active]=np.clip(coefficients['current_v4'][:,active]+coefficients['teacher_anger'][:,active]-coefficients['teacher_neutral'][:,active],0,1)
    hybrid[:,[15,16,21,22]]=np.clip(coefficients['teacher_anger'][:,[15,16,21,22]]-coefficients['teacher_neutral'][:,[15,16,21,22]],0,1)
    coefficients['hybrid_target']=hybrid
    for label,w in coefficients.items():
        surface[label+'_fit']=np.einsum('fc,cvk->fvk',w,fitter.shape[:,region],optimize=False)*10
        delta=np.einsum('fc,cpk->fpk',w,fitter.shape[:,landmarks],optimize=False)
        points[label+'_fit']=np.einsum('fpk,pk->fp',delta,direction,optimize=False)*10
    brow_only={k:np.einsum('fc,cvk->fvk',w[:,brows],fitter.shape[brows][:,region],optimize=False)*10
               for k,w in coefficients.items()}
    report=dict(sample_id=sample_id,text=sample['text'],audio_source=str(source),audio_sha256=sha(source),teacher_sha256=MODEL_SHA256,candidate_sha256=sha(NEW/'fullface.onnx'),
        controls=dict(emotion='anger',intensity=1.,gaze_style='Claire'),
        method='512 strongest canonical brow-support vertices; XYZ displacement RMS in mm. Hann periodogram after linear detrending, ignoring first/last 250 ms. No model or renderer changes.',
        surface={k:temporal_metrics(v) for k,v in surface.items()},
        brow_only_surface={k:temporal_metrics(v) for k,v in brow_only.items()},
        coefficients={k:temporal_metrics(v[:,brows]) for k,v in coefficients.items()},
        seed_difference_surface_rms_mm=float(np.sqrt(np.mean((surface['teacher_anger_raw']-surface['teacher_anger_seed1_raw'])**2))))
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'tracks.npz',**{k+'_weights':v for k,v in coefficients.items()},**{k+'_points':v for k,v in points.items()},brow_vertices=region,landmarks=landmarks)
    # The high-pass traces are diagnostic zero-phase filters, not a runtime fix.
    keys=['teacher_anger_raw','teacher_anger_fit','hybrid_target_fit','candidate_fit','current_v4_fit']
    labels=['NVIDIA raw geometry','NVIDIA after coefficient fit','Paired training target','New candidate','Current v4']
    fig,axes=plt.subplots(3,1,figsize=(11,8),gridspec_kw=dict(height_ratios=[1,1,1.15]))
    time=np.arange(len(audio)*60//16000+1)/60
    for key,label in zip(keys,labels):
        y=points[key][:,0]
        axes[0].plot(np.arange(len(y))/60,y,label=label,alpha=.85)
        # Reflect-padded Fourier high-pass, for illustration only. Avoid SciPy's
        # SOS steady-state BLAS calculation, which raises spurious exception
        # flags on this host. The RMS metrics use the periodogram above.
        padded=np.pad(linear_detrend(y[:,None])[:,0],(120,120),mode='reflect')
        fft=np.fft.rfft(padded);fft[np.fft.rfftfreq(len(padded),1/60)<4]=0
        fast=np.fft.irfft(fft,n=len(padded))[120:-120]
        if not np.isfinite(fast).all():raise FloatingPointError('Nonfinite diagnostic trace')
        axes[1].plot(np.arange(len(y))/60,fast,label=label,alpha=.85)
    axes[0].set_ylabel('Brow-down landmark (mm)');axes[0].legend(ncol=2,fontsize=8)
    axes[1].set_ylabel('Above-4-Hz component (mm)');axes[1].set_xlim(.25,len(audio)/16000-.25)
    for ax in axes[:2]:ax.grid(alpha=.2);ax.set_xlabel('Audio time (s)')
    values=[report['surface'][k]['fast_4_15_hz_rms'] for k in keys]
    axes[2].barh(labels,values);axes[2].invert_yaxis();axes[2].set_xlabel('4–15 Hz RMS over the brow region (mm)')
    fig.suptitle(sample['label']+' · anger 100% · eyebrow vibration provenance')
    fig.tight_layout();fig.savefig(out/'brow-diagnostic.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',default='appointment')
    args=parser.parse_args()
    main(args.sample)

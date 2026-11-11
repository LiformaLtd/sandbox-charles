"""Trace phrase-ending corner motion through raw teacher, fit and candidate."""
import json
import os
import sys
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from fit import FullFaceFit,PAIRS
from infer_full import sequence
from infer import sequence as compact_sequence


def stats(points,start):
    points=points[start:];steps=np.diff(points,axis=0);acc=np.diff(steps,axis=0)
    magnitudes=np.linalg.norm(steps,axis=-1)
    frame=int(np.argmax(magnitudes)//2)+start+1
    return dict(step_rms_mm=float(np.sqrt(np.mean(steps**2))),acceleration_rms_mm=float(np.sqrt(np.mean(acc**2))),
        max_step_mm=float(magnitudes.max()),max_step_frame=frame,max_step_seconds=frame/60,
        peak_to_peak_xyz_mm=np.ptp(points,axis=0).tolist())


def main():
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
    model=APP/'public'/manifest['candidate']['url'].lstrip('/')
    if sha(model)!=manifest['candidate']['sha256']:raise ValueError('Changed candidate')
    fit=FullFaceFit(ASSETS,WORK/'arkit-student/james_arkit_v3.ma',APP)
    pairs=PAIRS[[0,-1]];corners=fit.shape[:,pairs].mean(2)
    neutral=fit.neutral[pairs].mean(1)
    sys.path.insert(0,str(TEACHER))
    from app.audio2face_v3 import load_backend,animate,MODEL_SHA256
    backend=load_backend(TEACHER/'models_cache/audio2face-v3/network.onnx','mlx')
    out=APP/'.local/mouth-tail-diagnostic';out.mkdir(exist_ok=True)
    report=dict(candidate_sha256=sha(model),teacher_sha256=MODEL_SHA256,
        method='Two inner mouth-corner lip-pair midpoints in James, physical XYZ displacement in mm. Last 0.8 seconds measured; plots cover last 1.2 seconds. Acoustic endpoint is a diagnostic RMS threshold, not a phoneme label.',cases=[])
    traces={};fig,axes=plt.subplots(len(manifest['samples']),2,figsize=(13,14))
    for row,sample in enumerate(manifest['samples']):
        source=APP/'public'/sample['pcm'].lstrip('/');audio=np.fromfile(source,'<f4')
        if sha(source)!=sample['sha256']:raise ValueError('Audio changed')
        base=compact_sequence(BASELINE,audio)
        frame_count=len(base);energy=np.array([np.sqrt(np.mean(audio[i:i+320]**2)) for i in range(0,len(audio),320)])
        threshold=max(10**(-55/20),energy.max()*10**(-35/20));active=np.flatnonzero(energy>threshold)
        endpoint=min(len(audio)/16000,(int(active[-1])+1)*.02)
        for column,emotion in enumerate(['neutral','anger']):
            control=np.zeros((1,30,10),np.float32)
            if emotion!='neutral':control[:,:,EMOTIONS.index(emotion)]=1
            def infer(inputs):return backend({**inputs,'emotion':control})
            raw=[];fitted=[]
            for _,geometry in animate(audio,infer,identity='James',seed=0):
                w,_=fit(geometry);fitted.append(w)
                skin=geometry[:,:72006].reshape(-1,24002,3)+fit.model_neutral
                raw.append((skin[:,pairs].mean(2)-neutral)*10)
            candidate=sequence(model,audio,emotion,1.,'James')['blendshapes']
            weights=dict(teacher_fit=np.concatenate(fitted),current_v4=base,candidate=candidate)
            points=dict(teacher_raw=np.concatenate(raw))
            for name,w in weights.items():points[name]=np.einsum('fc,cpk->fpk',w,corners,optimize=False)*10
            if any(len(p)!=frame_count for p in points.values()):raise ValueError('Timing mismatch')
            start=max(0,frame_count-48);metrics={name:stats(p,start) for name,p in points.items()}
            key=sample['id']+'--'+emotion
            for name,p in points.items():traces[key+'--'+name+'--points']=p
            for name,w in weights.items():traces[key+'--'+name+'--weights']=w
            # Zero padding is already used by the runner. Extending the silent
            # tail must preserve all original frames, and reveals the relaxation.
            extended=sequence(model,np.pad(audio,(0,16000)),emotion,1.,'James')['blendshapes']
            drift=float(np.abs(candidate-extended[:frame_count]).max())
            traces[key+'--extended--points']=np.einsum('fc,cpk->fpk',extended,corners,optimize=False)*10
            maxframe=metrics['candidate']['max_step_frame'];joined=(maxframe-15)%30==0
            channels={fit.names[c]:dict(last_value=float(candidate[-1,c]),last_second_range=float(np.ptp(candidate[max(0,frame_count-60):,c])),
                step_at_largest_corner_jump=float(candidate[maxframe,c]-candidate[maxframe-1,c])) for c in range(14,41)}
            report['cases'].append(dict(sample=sample['id'],text=sample['text'],emotion=emotion,frames=frame_count,
                acoustic_endpoint_seconds=endpoint,trailing_seconds=len(audio)/16000-endpoint,
                tail=metrics,largest_candidate_step_at_window_join=joined,appended_silence_original_frame_drift=drift,channels=channels))
            ax=axes[row,column]
            for name,label in [('teacher_raw','Raw teacher'),('teacher_fit','Teacher fit'),('current_v4','Released v4'),('candidate','Candidate')]:
                ax.plot(np.arange(frame_count)/60,points[name][:,:,1].mean(1),label=label)
            ax.axvline(endpoint,color='grey',ls=':',lw=1)
            ax.set_xlim(max(0,frame_count/60-1.2),frame_count/60);ax.set_title(sample['id']+' · '+emotion,fontsize=9)
            ax.set_ylabel('Mean corner Y movement (mm)');ax.grid(alpha=.2)
            print(key, 'tail step raw/fit/v4/candidate', [round(metrics[n]['step_rms_mm'],4) for n in ['teacher_raw','teacher_fit','current_v4','candidate']],
                'largest at',round(maxframe/60,3),'join',joined,'extension drift',drift,flush=True)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'mouth-tails.png',dpi=150);plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'traces.npz',**traces)

if __name__=='__main__':main()

"""Test central-window mouth inference without altering upper-face tracks."""
import json
import numpy as np
import onnxruntime as ort
from common import *
from infer_full import sequence
from fit import PAIRS


def dense_mouth(session,pcm,prior,emotion,hop):
    n=len(prior);result=prior.copy();emotion_input=np.zeros((1,60,10),np.float32)
    if emotion!='neutral':emotion_input[:,:,EMOTIONS.index(emotion)]=1
    style=np.zeros((1,60,3),np.float32);style[:,:,1]=1
    if hop==1:
        windows=[(i*16000//60-8000,[30],[i]) for i in range(n)]
    else:
        left=30-hop//2;windows=[]
        for offset in range(-30,n-left,hop):
            local=np.arange(left,left+hop);global_frames=offset+local;valid=(global_frames>=0)&(global_frames<n)
            if valid.any():windows.append((offset*16000//60,local[valid],global_frames[valid]))
    for start,local,global_frames in windows:
        x=np.zeros((1,16000),np.float32);lo,hi=max(0,start),min(len(pcm),start+16000)
        if hi>lo:x[0,lo-start:hi-start]=pcm[lo:hi]
        p=session.run(['blendshapes'],dict(pcm=x,emotion=emotion_input,gaze_style=style))[0][0]
        result[np.asarray(global_frames),14:41]=p[np.asarray(local),14:41]
    return result


def main():
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text());model=APP/'public'/manifest['candidate']['url'].lstrip('/')
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    with np.load(OUT/'calibration.npz') as z:shape=z['shape'];gap=z['gap_basis'];rest=z['gap_neutral'];neutral=z['neutral']
    xyz=neutral;chin=np.flatnonzero((np.abs(xyz[:,0])<2)&(xyz[:,1]>162)&(xyz[:,1]<166)&(xyz[:,2]>3))
    teacher=np.load(APP/'.local/mouth-tail-diagnostic/traces.npz')
    references=np.load(APP/'.local/mouth-corner-isolated-v1/raw-gap-references.npz')
    corner=shape[:,PAIRS[[0,-1]]].mean(2);out=APP/'.local/center-mouth-probe';out.mkdir(exist_ok=True);rows=[];tracks={}
    for hop in [15,6,1]:
        for sample in manifest['samples']:
            pcm=np.fromfile(APP/'public'/sample['pcm'].lstrip('/'),'<f4')
            for emotion in ['neutral','anger']:
                original=sequence(model,pcm,emotion)['blendshapes'];new=dense_mouth(session,pcm,original,emotion,hop);key=sample['id']+'--'+emotion
                a=np.einsum('fc,ck->fk',original,gap,optimize=False)+rest;b=np.einsum('fc,ck->fk',new,gap,optimize=False)+rest
                old_rms=float(np.sqrt(np.mean((a-references[key])**2)))*10;new_rms=float(np.sqrt(np.mean((b-references[key])**2)))*10
                old_chin=np.einsum('fc,cvk->fvk',original[-13:],shape[:,chin],optimize=False)*10;new_chin=np.einsum('fc,cvk->fvk',new[-13:],shape[:,chin],optimize=False)*10
                old_step=float(np.sqrt(np.mean(np.diff(old_chin,axis=0)**2)));new_step=float(np.sqrt(np.mean(np.diff(new_chin,axis=0)**2)))
                corners=np.einsum('fc,cpk->fpk',new,corner,optimize=False)*10;truth=teacher[key+'--teacher_raw--points'];corner_error=float(np.sqrt(np.mean((corners[-48:]-truth[-48:])**2)))
                row=dict(hop=hop,sample=sample['id'],emotion=emotion,old_lip_rms_mm=old_rms,new_lip_rms_mm=new_rms,lip_ratio=new_rms/old_rms,
                    old_chin_step_mm=old_step,new_chin_step_mm=new_step,chin_step_ratio=new_step/max(old_step,1e-12),corner_error_mm=corner_error)
                rows.append(row);tracks[str(hop)+'--'+key]=new
                print(hop,sample['id'],emotion,'lip',round(row['lip_ratio'],3),'chin',round(row['chin_step_ratio'],3),'corner',round(corner_error,3),flush=True)
        pcm=np.zeros(32000,np.float32);prior=sequence(model,pcm)['blendshapes'];new=dense_mouth(session,pcm,prior,'neutral',hop)
        p=np.einsum('fc,cvk->fvk',new,shape[:,chin],optimize=False)*10;print('SILENCE',hop,'chin step',float(np.sqrt(np.mean(np.diff(p[30:],axis=0)**2))),flush=True)
    (out/'metrics.json').write_text(json.dumps(dict(model_sha256=sha(model),cases=rows),indent=2));np.savez_compressed(out/'tracks.npz',**tracks)

if __name__=='__main__':main()

"""Portable CPU runner for the full-face research graph and articulation track."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from infer import read_audio

EMOTIONS=['amazement','anger','cheekiness','disgust','fear','grief','joy','outofbreath','pain','sadness']
STYLES=['Claire','James','Mark']


def sequence(model,pcm,emotion='neutral',intensity=1.,gaze_style='James'):
    pcm=np.asarray(pcm,np.float32)
    if pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all() or np.max(np.abs(pcm))>1:
        raise ValueError('Expected finite nonempty normalized mono 16 kHz audio')
    if emotion not in ['neutral',*EMOTIONS] or gaze_style not in STYLES or not 0<=intensity<=1:
        raise ValueError('Invalid expression or gaze controls')
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    if {i.name for i in session.get_inputs()}!={'pcm','emotion','gaze_style'}:
        raise ValueError('Unexpected full-face model inputs')
    emotion_input=np.zeros((1,60,10),np.float32)
    if emotion!='neutral':emotion_input[:,:,EMOTIONS.index(emotion)]=intensity
    style=np.zeros((1,60,3),np.float32);style[:,:,STYLES.index(gaze_style)]=1
    outputs={v.name:[] for v in session.get_outputs()}
    for start in range(-8000,len(pcm)-4000,8000):
        window=np.zeros((1,16000),np.float32)
        lo,hi=max(0,start),min(len(pcm),start+16000)
        window[0,lo-start:hi-start]=pcm[lo:hi]
        result=session.run(None,dict(pcm=window,emotion=emotion_input,gaze_style=style))
        positions=start+np.arange(15,45)*16000//60
        valid=(positions>=0)&(positions<len(pcm))
        for name,value in zip(outputs,result):outputs[name].append(value[0,15:45][valid])
    tracks={name:np.concatenate(values) for name,values in outputs.items()}
    expected=(len(pcm)*60+15999)//16000
    if any(w.shape!=(expected,52) or not np.isfinite(w).all() or w.min()<0 or w.max()>1 for w in tracks.values()):
        raise ValueError('Incomplete or invalid full-face output')
    return tracks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ['model','audio','output']:parser.add_argument('--'+option,type=Path,required=True)
    parser.add_argument('--emotion',choices=['neutral',*EMOTIONS],default='neutral')
    parser.add_argument('--gaze-style',choices=STYLES,default='James')
    parser.add_argument('--intensity',type=float,default=1.)
    args=parser.parse_args()
    tracks=sequence(args.model,read_audio(args.audio),args.emotion,args.intensity,args.gaze_style)
    count=len(tracks['blendshapes'])
    np.savez_compressed(args.output,**tracks,sample_indices=np.arange(count)*16000//60,
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        audio_sha256=hashlib.sha256(args.audio.read_bytes()).hexdigest())
    print(json.dumps(dict(frames=count,outputs=list(tracks),output=str(args.output))))


if __name__=='__main__':main()

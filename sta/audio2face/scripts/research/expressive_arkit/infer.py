"""Explicit offline ONNX runner for the research model; no Player dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile


def read_audio(path):
    rate,pcm=wavfile.read(path)
    if rate!=16000 or pcm.ndim!=1 or not len(pcm):
        raise ValueError('Expected nonempty mono 16 kHz WAV')
    if pcm.dtype==np.int16:pcm=pcm.astype(np.float32)/32768
    elif pcm.dtype==np.int32:pcm=pcm.astype(np.float32)/2147483648
    elif pcm.dtype not in [np.float32,np.float64]:raise ValueError('Expected float or signed PCM16/PCM32 WAV')
    pcm=np.asarray(pcm,np.float32)
    if not np.isfinite(pcm).all() or np.max(np.abs(pcm))>1:raise ValueError('Invalid normalized audio')
    return pcm


def sequence(path,pcm,emotion=None):
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    names={v.name for v in session.get_inputs()}
    if names not in [{'pcm'},{'pcm','emotion'}]:raise ValueError('Unexpected model interface')
    if emotion is None:emotion=np.zeros(10,np.float32)
    emotion=np.asarray(emotion,np.float32)
    if emotion.shape!=(10,) or not np.isfinite(emotion).all() or np.min(emotion)<0 or np.max(emotion)>1:
        raise ValueError('Expected ten finite emotion intensities in [0,1]')
    output=[]
    for start in range(-8000,len(pcm)-4000,8000):
        audio=np.zeros((1,16000),np.float32)
        lo,hi=max(0,start),min(len(pcm),start+16000)
        audio[0,lo-start:hi-start]=pcm[lo:hi]
        inputs={'pcm':audio}
        if 'emotion' in names:inputs['emotion']=np.broadcast_to(emotion,(1,60,10)).copy()
        prediction=session.run(None,inputs)[0]
        positions=start+np.arange(15,45)*16000//60
        keep=(positions>=0)&(positions<len(pcm))
        output.append(prediction[0,15:45][keep])
    coefficients=np.concatenate(output)
    if coefficients.shape!=((len(pcm)*60+15999)//16000,52) or not np.isfinite(coefficients).all():
        raise ValueError('Incomplete inference track')
    return coefficients


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--emotion',default='neutral',choices=['neutral','amazement','anger','cheekiness','disgust','fear','grief','joy','outofbreath','pain','sadness'])
    parser.add_argument('--intensity',type=float,default=1.)
    args=parser.parse_args()
    emotions=['amazement','anger','cheekiness','disgust','fear','grief','joy','outofbreath','pain','sadness']
    control=np.zeros(10,np.float32)
    if args.emotion!='neutral':control[emotions.index(args.emotion)]=args.intensity
    w=sequence(args.model,read_audio(args.audio),control)
    np.savez_compressed(args.output,weights=w,sample_indices=np.arange(len(w))*16000//60,
        emotion=control,model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        audio_sha256=hashlib.sha256(args.audio.read_bytes()).hexdigest())
    print(json.dumps(dict(frames=len(w),fps=60,output=str(args.output))))


if __name__=='__main__':main()

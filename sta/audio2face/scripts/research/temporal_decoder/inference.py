"""Complete-utterance native inference with explicit valid-frame history."""
import numpy as np
OFFSETS=np.arange(15,45)*16000//60


def infer(session,audio):
    temporal=any(v.name=='history' for v in session.get_inputs())
    history=np.zeros((1,7,96),np.float32);frames=[]
    for start in range(-8000,len(audio)-4000,8000):
        valid=(start+OFFSETS>=0)&(start+OFFSETS<len(audio))
        if not valid.any():continue
        first=int(np.flatnonzero(valid)[0]+15);count=int(valid.sum())
        pcm=np.zeros((1,16000),np.float32);lo,hi=max(start,0),min(start+16000,len(audio));pcm[0,lo-start:hi-start]=audio[lo:hi]
        if temporal:
            track,history=session.run(None,dict(pcm=pcm,history=history,first=np.array([first],np.int64),count=np.array([count],np.int64)))
            frames.append(track[0,:count])
        else:frames.append(session.run(None,{'pcm':pcm})[0][0,first:first+count])
    result=np.concatenate(frames)
    if not np.isfinite(result).all():raise ValueError('Invalid inference')
    return result

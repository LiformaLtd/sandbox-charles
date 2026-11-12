import numpy as np
OFFSETS=np.arange(15,45)*16000//60

def infer(session,audio):
 recurrent=any(i.name=='state' for i in session.get_inputs());state=np.zeros((1,32),np.float32);pieces=[]
 for start in range(-8000,len(audio)-4000,8000):
  valid=(start+OFFSETS>=0)&(start+OFFSETS<len(audio))
  if not valid.any():continue
  first=int(np.flatnonzero(valid)[0]+15);count=int(valid.sum());pcm=np.zeros((1,16000),np.float32);lo,hi=max(start,0),min(start+16000,len(audio));pcm[0,lo-start:hi-start]=audio[lo:hi]
  if recurrent:
   out,state=session.run(None,dict(pcm=pcm,state=state,first=np.array([first],np.int64),count=np.array([count],np.int64)));pieces.append(out[0,:count])
  else:pieces.append(session.run(None,{'pcm':pcm})[0][0,first:first+count])
 result=np.concatenate(pieces);assert len(result)==(len(audio)*60+15999)//16000 and np.isfinite(result).all()
 return result

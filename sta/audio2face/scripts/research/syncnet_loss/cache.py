"""Training-only 25fps audio targets and exact60→25 coefficient interpolation."""
from common import *
import numpy as np,torch
import python_speech_features
from SyncNetModel import S

def main():
 torch.set_num_threads(8);net=S().eval();net.load_state_dict(torch.load(EXPERT/'vendor/syncnet_v2.model',map_location='cpu',weights_only=True));net.requires_grad_(False);maps=[];audio_targets=[];locks={};cursor=0
 for name in ['old-train','new-train']:
  root=SCALE/'datasets'/name;rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='train'];valid=np.load(SCALE/'features'/name/'valid.npy',mmap_mode='r');start_cursor=cursor;local=0
  for row in rows:
   pcm=np.fromfile(row['audio'],'<f4');assert sha(row['audio'])==row['sha256_audio'];mfcc=python_speech_features.mfcc(np.round(np.clip(pcm,-1,1)*32767).astype(np.int16),16000).T.astype(np.float32);locks[row['audio']]=row['sha256_audio'];candidates=[];map_rows=[]
   for start in range(-8000,len(pcm)-4000,8000):
    pos=start+np.arange(15,45)*16000//60;mask=valid[local];lo=np.flatnonzero(mask)[0];hi=np.flatnonzero(mask)[-1];qs=[]
    for q in range(max(0,int(np.ceil(pos[lo]/640))),int(pos[hi]/640)+1):
     times=(q+np.arange(5))*640
     if times[-1]>pos[max(lo,hi-1)] or (q+1)*4+20>mfcc.shape[1]:continue
     left=np.searchsorted(pos,times,side='right')-1
     if left.min()<0 or left.max()>=29 or not np.all(mask[left]&mask[left+1]):continue
     fraction=(times-pos[left])/(pos[left+1]-pos[left]);qs.append((q,left,fraction))
    if qs:
     q,left,fraction=min(qs,key=lambda v:abs(v[0]*640-(start+.42*16000)));map_rows.append(dict(eligible=True,left=left.tolist(),fraction=fraction.tolist(),q=int(q),source=row['stem']));candidates.append(mfcc[:,(q+1)*4:(q+1)*4+20])
    else:map_rows.append(dict(eligible=False,left=[0]*5,fraction=[0.]*5,q=-1,source=row['stem']));candidates.append(np.zeros((13,20),np.float32))
    local+=1;cursor+=1
   with torch.inference_mode():f=net.forward_aud(torch.tensor(np.stack(candidates)[:,None])).numpy()
   audio_targets.append(f);maps.extend(map_rows)
  assert local==len(valid);print(name,'mapped',local,'windows',flush=True)
 np.save(OUT/'audio-features.npy',np.concatenate(audio_targets));freeze(OUT/'aux-map.json',maps);freeze(OUT/'cache-lock.json',dict(windows=len(maps),eligible=sum(r['eligible'] for r in maps),audio_sha256=sha(OUT/'audio-features.npy'),map_sha256=sha(OUT/'aux-map.json'),waveforms=locks,sync_checkpoint_sha256=sha(EXPERT/'vendor/syncnet_v2.model'),source_sha256=sha(__file__),alignment='video q..q+4 vs audio MFCC(q+1)*4:(q+1)*4+20; fixed−40ms'))
if __name__=='__main__':main()

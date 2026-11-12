"""Frozen paired discrimination on James; no expert/student adaptation."""
from common import *
import cv2,numpy as np,torch,time
from expert import Expert
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask
sys.path.insert(0,str(OUT/'vendor-deps'));sys.path.insert(0,str(VENDOR/'syncnet'))
import python_speech_features
from SyncNetModel import S
KINDS=['correct','shift-m8','shift-m4','shift-m2','shift-p2','shift-p4','shift-p8','underopen-50','missing-closure','static']
MODES={'paper-av':(.4,False),'checkpoint-av':(.165,False),'paper-muted':(.4,True)}

def images(case,kind,view):
 root=OUT/'frames'/case['id']/kind/view
 return np.stack([cv2.cvtColor(cv2.imread(str(root/f'{t:04}.png')),cv2.COLOR_BGR2RGB) for t in range(case['frames'])])
def movie(base,kind):
 if kind.startswith('shift-'):
  sign=-1 if kind[6]=='m' else 1;delta=sign*int(kind[7:]);return base[np.clip(np.arange(len(base))+delta,0,len(base)-1)]
 if kind=='static':return np.repeat(base[len(base)//2:len(base)//2+1],len(base),axis=0)
 return base

@torch.inference_mode()
def avloss(expert,vfeat,afeat,target):
 net=expert.net;x=net.fusion(torch.cat((vfeat,afeat),-1));length=torch.tensor([x.shape[1]]);ctc,_=net.ctc(x,length,target)
 if net.proj_decoder:x=net.proj_decoder(x)
 yin,yout=add_sos_eos(target,net.sos,net.eos,net.ignore_id);pred,_=net.decoder(yin,target_mask(yin,net.ignore_id),x,torch.ones((1,1,x.shape[1]),dtype=torch.bool));att=net.criterion(pred,yout);loss=net.mtlalpha*ctc+(1-net.mtlalpha)*att
 return dict(loss=float(loss),ctc=float(ctc),attention=float(att))

class Sync:
 def __init__(self):
  self.net=S().eval();self.net.load_state_dict(torch.load(VENDOR/'syncnet_v2.model',map_location='cpu',weights_only=True));self.net.requires_grad_(False)
 @torch.inference_mode()
 def features(self,frames,pcm):
  # Full-face crop pinned from James setup geometry, no face-detector dependency.
  bgr=np.stack([cv2.resize(f[16:176,32:192,::-1],(224,224),interpolation=cv2.INTER_LINEAR) for f in frames]).astype(np.float32)
  n=len(frames)-5;v=[]
  for start in range(0,n,8):
   x=np.stack([bgr[t:t+5].transpose(3,0,1,2) for t in range(start,min(n,start+8))]);v.append(self.net.forward_lip(torch.from_numpy(x)).numpy())
  wav=np.round(np.clip(pcm,-1,1)*32767).astype(np.int16);mfcc=python_speech_features.mfcc(wav,16000).T.astype(np.float32);a=[]
  for start in range(0,n,32):
   x=np.stack([mfcc[:,t*4:t*4+20] for t in range(start,min(n,start+32))])[:,None];a.append(self.net.forward_aud(torch.from_numpy(x)).numpy())
  return np.concatenate(v),np.concatenate(a)

def main():
 p=json.loads((OUT/'protocol.json').read_text())
 for name,digest in p['sources'].items():assert sha(APP/name)==digest,name
 for name,digest in p['weights'].items():assert sha(VENDOR/name)==digest,name
 assert sha(OUT/'cases.json')==p['cases_sha256'];cases=json.loads((OUT/'cases.json').read_text());expert=Expert();sync=Sync();began=time.monotonic();avrows=[];sync_features={};controls=[]
 for case in cases:
  pcm=np.fromfile(OUT/'audio'/(case['id']+'.f32'),'<f4');base=images(case,'correct','mouth');face=images(case,'correct','face');n=case['frames'];target=expert.target(case['text']);audio=expert.audio(pcm,n);zero=expert.audio(np.zeros_like(pcm),n);mask=torch.ones((1,1,n),dtype=torch.bool)
  with torch.inference_mode():a,_=expert.net.aux_encoder(audio,mask);az,_=expert.net.aux_encoder(zero,mask)
  visual_cache={};vcorrect,afeat=sync.features(face,pcm);sync_features[case['id']]={'audio':afeat,'correct':vcorrect}
  for kind in KINDS:
   if case['split']=='calibration' and kind!='correct':continue
   if kind in ['underopen-50','missing-closure']:frames=images(case,kind,'mouth');f=images(case,kind,'face');v,_=sync.features(f,pcm);sync_features[case['id']][kind]=v
   else:frames=movie(base,kind)
   if kind=='static':v,_=sync.features(movie(face,kind),pcm);sync_features[case['id']][kind]=v
   for mode,(std,muted) in MODES.items():
    key=(kind,std)
    with torch.inference_mode():
     if key not in visual_cache:visual_cache[key]=expert.net.encoder(expert.video(frames,std),mask)[0]
     score=avloss(expert,visual_cache[key],az if muted else a,target)
    assert all(np.isfinite(v) for v in score.values());avrows.append(dict(case=case['id'],split=case['split'],engine=case['engine'],voice=case['voice'],kind=kind,mode=mode,**score))
   if kind=='correct':
    with torch.inference_mode():repeat=avloss(expert,visual_cache['correct',.4],a,target);wrong=avloss(expert,visual_cache['correct',.4],a,target.flip(-1))
    direct=expert.score(base,pcm,case['text']);assert abs(repeat['loss']-direct['loss'])<1e-6;controls.append(dict(case=case['id'],repeat_delta=abs(repeat['loss']-direct['loss']),correct=repeat['loss'],reversed_token_loss=wrong['loss']))
  np.savez(OUT/'scores'/(case['id']+'-sync.npz'),**sync_features[case['id']]);freeze(OUT/'scores'/(case['id']+'-av.json'),[r for r in avrows if r['case']==case['id']]);print(case['id'],case['split'],'scored; seconds',round(time.monotonic()-began),flush=True)
 # Only calibration correct tracks choose one global SyncNet offset. No per-variant re-alignment.
 offsets={}
 for offset in range(-5,6):
  d=[]
  for case in cases[:4]:
   z=sync_features[case['id']];t=np.arange(18,len(z['audio'])-18);d.append(float(np.linalg.norm(z['correct'][t+offset]-z['audio'][t],axis=-1).mean()))
  offsets[offset]=float(np.mean(d))
 offset=min(offsets,key=lambda k:(offsets[k],abs(k),k));freeze(OUT/'sync-calibration.json',dict(offset_frames=offset,curve=offsets,calibration_case_ids=[r['id'] for r in cases[:4]]));srows=[]
 for case in cases[4:]:
  z=sync_features[case['id']];t=np.arange(18,len(z['audio'])-18)
  for kind in KINDS:
   for label,o in [('zero',0),('calibrated',offset)]:
    delta=(-1 if kind[6]=='m' else 1)*int(kind[7:]) if kind.startswith('shift-') else 0;key='correct' if kind.startswith('shift-') else kind;idx=t+o+delta
    score=float(np.linalg.norm(z[key][idx]-z['audio'][t],axis=-1).mean());srows.append(dict(case=case['id'],engine=case['engine'],voice=case['voice'],kind=kind,mode=label,distance=score,windows=len(t)))
 freeze(OUT/'av-scores.json',avrows);freeze(OUT/'sync-scores.json',srows);freeze(OUT/'controls.json',controls);freeze(OUT/'completion.json',dict(seconds=time.monotonic()-began,av_rows=len(avrows),sync_rows=len(srows),cases=len(cases),source_sha256=sha(__file__)));print('Complete',flush=True)
if __name__=='__main__':main()

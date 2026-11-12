"""Independent frozen-renderer corruption check; no scorer realignment."""
from common import *
from render import Renderer
from SyncNetModel import S
import numpy as np,torch,python_speech_features,time

def main():
 torch.set_num_threads(8);net=S().eval();net.load_state_dict(torch.load(EXPERT/'vendor/syncnet_v2.model',map_location='cpu',weights_only=True));net.requires_grad_(False);renderer=Renderer();root=EXPERT/'confirmation';cases=json.loads((root/'cases.json').read_text());rows=[];began=time.monotonic()
 prior=json.loads((root/'protocol.json').read_text())
 for name,digest in prior['files'].items():assert sha(root/name)==digest,name
 freeze(OUT/'renderer-confirmation-protocol.json',dict(renderer_sha256=sha(HERE/'render.py'),common_sha256=sha(HERE/'common.py'),source_sha256=sha(__file__),cases_sha256=sha(root/'cases.json'),alignment_ms=-40,minimum_wins=6,clips=8))
 for case in cases:
  n=case['frames'];features={};pcm=np.fromfile(root/'audio'/(case['id']+'.f32'),'<f4');mfcc=python_speech_features.mfcc(np.round(np.clip(pcm,-1,1)*32767).astype(np.int16),16000).T.astype(np.float32)
  with torch.inference_mode():
   audio=net.forward_aud(torch.tensor(np.stack([mfcc[:,i*4:i*4+20] for i in range(n-5)])[:,None])).numpy()
   for kind in ['correct','underopen-50','missing-closure']:
    w=np.fromfile(root/'tracks'/(case['id']+'-'+kind+'.f32'),'<f4').reshape(n,52);video=torch.cat([renderer(torch.tensor(w[i:i+5])) for i in range(0,n,5)]);features[kind]=torch.cat([net.forward_lip(torch.stack([video[t:t+5].permute(1,0,2,3) for t in range(i,min(n-5,i+8))])) for i in range(0,n-5,8)]).numpy()
  t=np.arange(18,len(audio)-18)
  for kind in ['correct','underopen-50','missing-closure']+[f'shift-{s}{k}' for k in [2,4,8] for s in ['m','p']]:
   delta=(-1 if kind[6]=='m' else 1)*int(kind[7:]) if kind.startswith('shift-') else 0;key='correct' if kind.startswith('shift-') else kind;rows.append(dict(case=case['id'],engine=case['engine'],kind=kind,distance=float(np.linalg.norm(features[key][t-1+delta]-audio[t],axis=-1).mean())))
  print(case['id'],'seconds',round(time.monotonic()-began),flush=True)
 base={r['case']:r['distance'] for r in rows if r['kind']=='correct'};kinds=sorted({r['kind'] for r in rows}-{'correct'});wins={k:sum(r['distance']>base[r['case']]+1e-5 for r in rows if r['kind']==k) for k in kinds};freeze(OUT/'renderer-confirmation.json',dict(rows=rows,wins=wins,passes=bool(min(wins.values())>=6),seconds=time.monotonic()-began));print(wins,flush=True)
if __name__=='__main__':main()

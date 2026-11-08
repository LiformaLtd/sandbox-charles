"""Freeze and screen the offline renderer before using any auxiliary training loss."""
from common import *
import numpy as np,torch,time
from render import Renderer
from SyncNetModel import S
KINDS=['correct','underopen-50','missing-closure']

def main():
 torch.set_num_threads(8);torch.manual_seed(37);net=S().eval();net.load_state_dict(torch.load(EXPERT/'vendor/syncnet/syncnet_v2.model',map_location='cpu',weights_only=True)) if False else net.load_state_dict(torch.load(EXPERT/'vendor/syncnet_v2.model',map_location='cpu',weights_only=True));net.requires_grad_(False);renderer=Renderer()
 cases=json.loads((EXPERT/'cases.json').read_text());freeze(OUT/'renderer-protocol.json',dict(sources={str(p.relative_to(APP)):sha(p) for p in [HERE/'common.py',HERE/'render.py',HERE/'preflight.py']},neutral_photo_sha256=sha(OUT/'neutral-bgr.npy'),source_cases_sha256=sha(EXPERT/'cases.json'),sync_checkpoint_sha256=sha(EXPERT/'vendor/syncnet_v2.model'),method='CPU smooth compact point-splat James mouth patch over fixed Three neutral background; renderer differs from browser; no expert training'))
 rows=[];began=time.monotonic()
 for case in cases:
  features={};n=case['frames']
  for kind in KINDS:
   w=np.fromfile(EXPERT/'tracks'/(case['id']+'-'+kind+'.f32'),'<f4').reshape(n,52);renders=[]
   with torch.inference_mode():
    for i in range(0,n,5):renders.append(renderer(torch.tensor(w[i:i+5])))
    video=torch.cat(renders);features[kind]=torch.cat([net.forward_lip(torch.stack([video[t:t+5].permute(1,0,2,3) for t in range(i,min(n-5,i+8))])) for i in range(0,n-5,8)]).numpy()
  audio=np.load(EXPERT/'scores'/(case['id']+'-sync.npz'))['audio'];t=np.arange(18,len(audio)-18)
  for kind in KINDS+['shift-m4','shift-p4','shift-m8','shift-p8']:
   delta=(-1 if kind[6]=='m' else 1)*int(kind[7:]) if kind.startswith('shift-') else 0;key='correct' if kind.startswith('shift-') else kind;distance=float(np.linalg.norm(features[key][t-1+delta]-audio[t],axis=-1).mean());rows.append(dict(case=case['id'],split=case['split'],engine=case['engine'],kind=kind,distance=distance))
  (OUT/'renderer-progress.json').write_text(json.dumps(rows,indent=2));print(case['id'],case['split'],'seconds',round(time.monotonic()-began),flush=True)
 test=[r for r in rows if r['split']=='test'];base={r['case']:r['distance'] for r in test if r['kind']=='correct'};wins={k:sum(r['distance']>base[r['case']]+1e-5 for r in test if r['kind']==k) for k in KINDS[1:]+['shift-m4','shift-p4','shift-m8','shift-p8']};freeze(OUT/'renderer-compatibility.json',dict(rows=rows,wins=wins,passes=bool(min(wins.values())>=6),seconds=time.monotonic()-began));print(wins,flush=True)
 # Compare analytic to independent central differences of the actual rendered SyncNet score.
 case=cases[0];w=np.fromfile(EXPERT/'tracks'/(case['id']+'-correct.f32'),'<f4').reshape(-1,52)[30:35].copy();a=np.load(EXPERT/'scores'/(case['id']+'-sync.npz'))['audio'][31];net=net.double();x=torch.tensor(w,dtype=torch.float64,requires_grad=True)
 def f(v):return torch.linalg.vector_norm(net.forward_lip(renderer(v).permute(1,0,2,3)[None])[0]-torch.tensor(a))
 value=f(x);grad=torch.autograd.grad(value,x)[0];checks=[]
 for channel in [17,20,21]:
  direction=torch.zeros_like(x);direction[:,channel]=1;epsilon=.000001;plus=float(f(x.detach()+epsilon*direction));minus=float(f(x.detach()-epsilon*direction));numeric=(plus-minus)/(2*epsilon);analytic=float((grad*direction).sum());relative=abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-4);checks.append(dict(channel=channel,analytic=analytic,numeric=numeric,relative=relative))
 freeze(OUT/'renderer-gradient.json',dict(loss=float(value),gradient_norm=float(grad.norm()),checks=checks,all_finite=bool(torch.isfinite(grad).all()),passes=bool(torch.isfinite(grad).all() and all(c['relative']<.02 for c in checks)),numeric_precision='float64 renderer/expert central differences epsilon1e-6',source_sha256=sha(__file__)));print(checks,flush=True)
if __name__=='__main__':main()

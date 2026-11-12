"""Post-screen confirmation: new corpus/texts, same corruptions and fixed expert."""
from common import *
import numpy as np
from prepare import variants,runs,associate
CONFIRM=OUT/'confirmation';SOURCE=APP/'.local/compact-feature-v1/confirmation/variants'

def main():
 prior=json.loads((OUT/'cases.json').read_text());seen={r['text'] for r in prior};rig=np.load(WORK/'arkit-student/rig-metrics.npz');B=rig['gap_basis'];rest=rig['rest']+rig['gap_offset'];names=json.loads((WORK/'arkit-student/channels.json').read_text());jaw=names.index('jawOpen');rows=[r for r in json.loads((SOURCE/'manifest.json').read_text()) if r['condition']=='original'];voices=sorted({r['engine']+':'+r['voice'] for r in rows});cases=[]
 for repeat in range(2):
  for voice in voices:
   choices=sorted([r for r in rows if r['engine']+':'+r['voice']==voice and r['text'] not in seen],key=lambda r:hashlib.sha256(('sync-confirm-101:'+str(repeat)+r['text']).encode()).hexdigest())
   for row in choices:
    pcm=np.fromfile(row['audio'],'<f4');n=len(pcm)//640
    if n<60:continue
    with np.load(SOURCE/'labels'/(row['stem']+'.npz')) as z:raw=z['weights'];rawg=z['gaps']
    t=np.arange(n)*60/25;left=np.minimum(t.astype(int),len(raw)-1);right=np.minimum(left+1,len(raw)-1);w=(raw[left]+(t-left)[:,None]*(raw[right]-raw[left])).astype(np.float32);w[:,[i for i,name in enumerate(names) if not name.startswith(('mouth','jaw'))]]=0
    gaps=np.einsum('tc,ck->tk',w,B,optimize=False)+rest;median=np.median(gaps,-1);closed=median<=.1;opened=median>.35
    if closed[10:-10].sum()<3 or opened.sum()<10:continue
    events=[r for r in associate(row['phones'],runs(np.median(rawg,-1)<=.1),len(rawg)) if r['status']=='unique'];mask=np.zeros(n,bool)
    for e in events:
     a,b=e['event'];mask|=(np.arange(n)/25>=a/60)&(np.arange(n)/25<b/60)&closed
    if mask.sum()<2:continue
    vv,mask=variants(w,gaps,jaw,B[jaw],mask);under=np.einsum('tc,ck->tk',vv['underopen-50'],B,optimize=False)+rest;missing=np.einsum('tc,ck->tk',vv['missing-closure'],B,optimize=False)+rest;reduction=(median-np.median(under,-1))[opened];lifted=np.median(missing,-1)[mask]
    if np.median(reduction)<.15 or np.min(lifted)<.3:continue
    break
   else:raise ValueError('Insufficient independent confirmation for '+voice)
   case=dict(id=f'confirm-{len(cases):02}',engine=row['engine'],voice=row['voice'],text=row['text'],source=row['stem'],audio_sha256=sha(row['audio']),label_sha256=sha(SOURCE/'labels'/(row['stem']+'.npz')),frames=n,targeted_closure_frames=int(mask.sum()),underopen_median_reduction_mm=float(10*np.median(reduction)),missing_min_gap_mm=float(10*np.min(lifted)))
   for kind,a in vv.items():freeze(CONFIRM/'tracks'/(case['id']+'-'+kind+'.f32'),a.astype('<f4').tobytes())
   freeze(CONFIRM/'audio'/(case['id']+'.f32'),pcm[:n*640].astype('<f4').tobytes());seen.add(row['text']);cases.append(case)
 freeze(CONFIRM/'cases.json',cases)
 files={str(p.relative_to(CONFIRM)):sha(p) for name in ['audio','tracks'] for p in (CONFIRM/name).iterdir()}
 freeze(CONFIRM/'protocol.json',dict(parent_protocol_sha256=sha(OUT/'protocol.json'),cases_sha256=sha(CONFIRM/'cases.json'),files=files,global_offset_frames=-1,sync_calibration_sha256=sha(OUT/'sync-calibration.json'),expert_selected_after_screen=True,required=['shift-m8','shift-m4','shift-p4','shift-p8','underopen-50','missing-closure'],threshold='6/8 and at least2/4 each engine, delta>1e-5; no recalibration',sources={str(p.relative_to(APP)):sha(p) for p in [HERE/'confirm_prepare.py',HERE/'confirm_score.py',HERE/'render.mjs',HERE/'score.py',HERE/'prepare.py']}))
 print('Frozen8 independent-corpus confirmation cases, unchanged expert/offset/corruptions.',flush=True)
if __name__=='__main__':main()

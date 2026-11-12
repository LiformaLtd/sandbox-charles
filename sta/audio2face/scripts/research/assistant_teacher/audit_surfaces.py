"""Supplement the fixed audit with full weighted skin residuals and ambiguity."""
from common import *
import numpy as np
sys.path.insert(0,str(APP/'scripts'))
from prepare_blendshapes import maya_nodes,maya_deltas,jaw_delta

def main():
    root=OUT/'audit';audit=json.loads((root/'result.json').read_text());protocol=json.loads((root/'protocol.json').read_text());assets=WORK/'v3-model'
    rig=np.load(assets/'bs_skin_James.npz');model=np.load(assets/'model_data_James.npz');names=[n.decode() for n in rig['poseNames'][1:]];active=np.flatnonzero(json.loads((assets/'bs_skin_config_James.json').read_text())['blendshape_params']['bsSolveActivePoses'])
    mask=rig['frontalMask'];skin=model['neutral_skin'];mouth=(np.abs(skin[:,0])<3)&(skin[:,1]>166.1)&(skin[:,1]<169)&(skin[:,2]>5);weights=np.repeat(np.where(mouth[mask],4.,1.).astype(np.float32),3)
    B=np.stack([rig[names[i]][mask].ravel() for i in active]).T;SG=B.T@(B*weights[:,None]);metrics=np.load(WORK/'arkit-student/rig-metrics.npz');L=metrics['gap_basis'][active];neutral_gap=metrics['rest']+metrics['gap_offset'];G=SG+np.eye(len(active),dtype=np.float32)+192*(L@L.T)
    nodes=maya_nodes((WORK/'arkit-student/james_arkit_v3.ma').read_text());neutral,teeth=maya_deltas(nodes,'c_bottomDenture_hi_bs','c_bottomDenture_hiShapeOrig',[names[i] for i in active]);C=np.stack([jaw_delta(neutral,teeth[names[i]],model['neutral_jaw']).ravel() if names[i] in teeth else np.zeros(15) for i in active]).astype(np.float32)
    sigma=audit['sigma_cm'];lam=audit['lambda0'];step=float(1/np.linalg.eigvalsh(G+lam*(C@C.T))[-1]);details=[]
    for row in protocol['choices']:
        with np.load(root/(row['stem']+'.npz')) as z:t={k:z[k] for k in z.files}
        j=np.linalg.norm(t['jaw'].reshape(-1,5,3).mean(1),axis=-1);strength=lam*np.exp(-j*j/(2*sigma*sigma))*(1-np.exp(-t['gaps'][:,4]**2/(2*sigma*sigma)))
        rhs=t['rhs']+strength[:,None]*(t['jaw']@C.T);x=t['weights'].copy();y=x.copy();momentum=1.
        for _ in range(95):
            new=np.clip(y-step*(y@G+strength[:,None]*((y@C)@C.T)-rhs),0,1);n=(1+np.sqrt(1+4*momentum**2))/2;y=new+np.float32((momentum-1)/n)*(new-x);x=new;momentum=n
        skin_rhs=t['rhs']-192*((t['gaps']-neutral_gap)@L.T);detail=dict(stem=row['stem'],split=row['split'],voice=row['engine']+':'+row['voice'])
        for arm,w in [('original',t['weights']),('jaw_soft',x)]:
            # Float64 avoids catastrophic cancellation in the small residual.
            w=w.astype(np.float64);sq=np.einsum('ti,ij,tj->t',w,SG.astype(np.float64),w)-2*(w*skin_rhs.astype(np.float64)).sum(-1)+t['skin_norm'].astype(np.float64)
            detail[arm]=dict(weighted_skin_rms_mm=float(10*np.sqrt(np.maximum(sq,0).sum()/(len(w)*weights.sum()))),max_coefficient_delta=float(np.max(np.abs(w-t['weights']))))
        details.append(detail)
    report={}
    for split in ['train','dev']:
        report[split]={a:float(np.mean([np.mean([d[a]['weighted_skin_rms_mm'] for d in details if d['split']==split and d['voice']==v]) for v in sorted({d['voice'] for d in details if d['split']==split})])) for a in ['original','jaw_soft']}
    freeze(root/'surface-check.json',dict(source_sha256=sha(__file__),audit_sha256=sha(root/'result.json'),weighted_skin_macro_rms_mm=report,clips=details,note='Weighted frontal surface; includes numerical clipping of tiny negative reconstructed squared residuals. Not a learnability test.'))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

"""Regenerate fixed clips; audit skin-only decomposition without replacing labels."""
from common import *
import importlib.util, numpy as np
sys.path.insert(0, str(TEACHER))
sys.path.insert(0, str(APP/'scripts'))
sys.path.insert(0, str(APP/'scripts/research/multivoice'))
from app.audio2face_v3 import animate, load_backend, DEFAULT_MODEL_PATH, MODEL_SHA256
from prepare_blendshapes import maya_nodes, maya_deltas, jaw_delta
import closure_metrics

def main():
    assets=WORK/'v3-model'; root=OUT/'audit';root.mkdir(parents=True,exist_ok=True)
    choices=[]
    for part,split in [('old-train','train'),('common-dev','dev')]:
        data=SCALE/'datasets'/part
        rows=[r for r in json.loads((data/'manifest.json').read_text()) if r['split']==split and r['condition']=='original']
        for voice in sorted({r['engine']+':'+r['voice'] for r in rows}):
            group=sorted([r for r in rows if r['engine']+':'+r['voice']==voice],key=lambda r:hashlib.sha256(('jaw-audit-37:'+r['stem']).encode()).hexdigest())[:2]
            choices.extend([dict(r,label_root=str(data/'labels')) for r in group])
    source=APP/'.local/compact-coverage-v2/coverage/sources/label.py'
    spec=importlib.util.spec_from_file_location('jaw_original_fitter',source);fitmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(fitmod)
    names,fit,_,geometry_gaps=fitmod.prepare_fit(assets)
    rig=np.load(assets/'bs_skin_James.npz');model=np.load(assets/'model_data_James.npz');active=np.flatnonzero(json.loads((assets/'bs_skin_config_James.json').read_text())['blendshape_params']['bsSolveActivePoses'])
    nodes=maya_nodes((WORK/'arkit-student/james_arkit_v3.ma').read_text());neutral_teeth,teeth=maya_deltas(nodes,'c_bottomDenture_hi_bs','c_bottomDenture_hiShapeOrig',[names[i] for i in active])
    C=np.stack([jaw_delta(neutral_teeth,teeth[names[i]],model['neutral_jaw']).ravel() if names[i] in teeth else np.zeros(15) for i in active]).astype(np.float32)
    mask=rig['frontalMask'];skin=model['neutral_skin'];mouth=(np.abs(skin[:,0])<3)&(skin[:,1]>166.1)&(skin[:,1]<169)&(skin[:,2]>5)
    weight=np.repeat(np.where(mouth[mask],4.,1.).astype(np.float32),3)
    B=np.stack([rig[names[i]][mask].ravel() for i in active]).T;offset=(skin-rig['neutral'])[mask].ravel();WB=B*weight[:,None]
    u,l=fitmod.LIP_PAIRS.T;L=np.stack([rig[names[i]][u,1]-rig[names[i]][l,1] for i in active]);neutral_gap=rig['neutral'][u,1]-rig['neutral'][l,1]
    G=B.T@WB+np.eye(len(active),dtype=np.float32)+192*(L@L.T)
    protocol=dict(source_sha256=sha(__file__),fitter_sha256=sha(source),teacher_sha256=MODEL_SHA256,scene_sha256=sha(WORK/'arkit-student/james_arkit_v3.ma'),choices=choices,assets={p.name:sha(p) for p in assets.glob('*James*')},jaw_term_trace_fraction=.1,iterations=95)
    freeze(root/'protocol.json',protocol)
    backend=None;tracks=[]
    for row in choices:
        cache=root/(row['stem']+'.npz')
        if not cache.exists():
            if backend is None:backend=load_backend(DEFAULT_MODEL_PATH,'mlx')
            if sha(row['audio'])!=row['sha256_audio']:raise ValueError('Changed audio')
            accum={k:[] for k in ['weights','gaps','jaw','rhs','skin_norm']}
            for indices,geo in animate(np.fromfile(row['audio'],'<f4'),backend,identity='James',seed=0):
                ws=fit(geo)[:,active];gap=geometry_gaps(geo);target=geo[:,:72006].reshape(-1,24002,3)[:,mask].reshape(len(geo),-1)+offset
                accum['weights'].append(ws);accum['gaps'].append(gap);accum['jaw'].append(geo[:,88812:88827]);accum['rhs'].append(target@WB+192*((gap-neutral_gap)@L.T));accum['skin_norm'].append((target**2*weight).sum(-1))
            np.savez(cache,**{k:np.concatenate(v) for k,v in accum.items()})
        with np.load(cache) as z:track={k:z[k].copy() for k in z.files}
        with np.load(Path(row['label_root'])/(row['stem']+'.npz')) as old:
            dw=float(np.max(np.abs(track['weights']-old['weights'][:,active])));dg=float(np.max(np.abs(track['gaps']-old['gaps'])))
            if dw>5e-5 or dg>5e-5:raise ValueError(('Teacher/fitter cache mismatch',row['stem'],dw,dg))
        track['row']=row;track['parity']=[dw,dg];tracks.append(track)
        print('regenerated',row['stem'],flush=True)
    sigma=float(np.median(np.concatenate([np.linalg.norm(t['jaw'].reshape(-1,5,3).mean(1),axis=-1) for t in tracks if t['row']['split']=='train'])))
    sigma=max(sigma,1e-4);lambda0=float(.1*np.trace(G)/np.trace(C@C.T));step=float(1/np.linalg.eigvalsh(G+lambda0*(C@C.T))[-1])
    conditions={}
    for split in ['train','dev']:
        scored={arm:[] for arm in ['original','jaw_soft']};details=[]
        for t in tracks:
            if t['row']['split']!=split:continue
            jaw=t['jaw'];jaw_norm=np.linalg.norm(jaw.reshape(-1,5,3).mean(1),axis=-1);lip=t['gaps'][:,4]
            strength=lambda0*np.exp(-jaw_norm**2/(2*sigma**2))*(1-np.exp(-lip**2/(2*sigma**2)))
            rhs=t['rhs']+strength[:,None]*(jaw@C.T);x=t['weights'].copy();y=x.copy();momentum=1.
            for _ in range(95):
                new=np.clip(y-step*(y@G+strength[:,None]*((y@C)@C.T)-rhs),0,1);n=(1+np.sqrt(1+4*momentum**2))/2;y=new+np.float32((momentum-1)/n)*(new-x);x=new;momentum=n
            report=dict(stem=t['row']['stem'],voice=t['row']['engine']+':'+t['row']['voice'],frames=len(jaw),cache_max_delta=t['parity'])
            for arm,w in [('original',t['weights']),('jaw_soft',x)]:
                gap=w@L+neutral_gap;actual=np.median(t['gaps'],-1)<=.1;pred=np.median(gap,-1)<=.1
                row=dict(t['row']);row['phones']=[dict(p,phone=p['phone'].lower()) for p in row['phones']]
                scored[arm].append((row,t['gaps'],gap));report[arm]=dict(gap_squared=float(np.square(gap-t['gaps']).sum()),gap_count=gap.size,jaw_squared=float(np.square(w@C-jaw).sum()),jaw_count=jaw.size,tp=int((actual&pred).sum()),actual=int(actual.sum()),pred=int(pred.sum()),weight_velocity=float(np.abs(np.diff(w,axis=0)).mean()))
            details.append(report)
        summary={}
        for arm,records in scored.items():
            voices={}
            for voice in sorted({d['voice'] for d in details}):
                d=[r[arm] for r in details if r['voice']==voice];voices[voice]=dict(gap_rms_mm=float(10*np.sqrt(sum(r['gap_squared'] for r in d)/sum(r['gap_count'] for r in d))),jaw_rms_mm=float(10*np.sqrt(sum(r['jaw_squared'] for r in d)/sum(r['jaw_count'] for r in d))),f1=float(2*sum(r['tp'] for r in d)/max(1,sum(r['actual']+r['pred'] for r in d))),weight_velocity=float(np.mean([r['weight_velocity'] for r in d])))
            summary[arm]=dict(voices=voices,macro={k:float(np.mean([v[k] for v in voices.values()])) for k in ['gap_rms_mm','jaw_rms_mm','f1','weight_velocity']},closures=closure_metrics.score(records,lead=.1)['summary'])
        conditions[split]=dict(summary=summary,clips=details)
    a,b=[list(active).index(names.index(n)) for n in ['jawOpen','mouthClose']]
    cosine=lambda x,y:float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))
    result=dict(protocol_sha256=sha(root/'protocol.json'),sigma_cm=sigma,lambda0=lambda0,skin_jawOpen_mouthClose_cosine=cosine(B[:,a]*np.sqrt(weight),B[:,b]*np.sqrt(weight)),gap_jawOpen_mouthClose_cosine=cosine(L[a],L[b]),jaw_nonzero_channels=[names[i] for i,c in zip(active,C) if np.max(np.abs(c))>1e-5],conditions=conditions,note='Exploratory fixed small audit. No target replacement, training change, smoothing or proof of learnability.')
    freeze(root/'result.json',result);print(json.dumps({s:r['summary'] for s,r in conditions.items()},indent=2),flush=True)

if __name__=='__main__':main()

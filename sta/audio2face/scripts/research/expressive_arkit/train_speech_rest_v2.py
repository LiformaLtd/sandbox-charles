"""Train mouth/chin settling with raw lip-gap and existing-corner guards."""
import copy
import json
import time
import numpy as np
import torch
from common import *
from speech_rest_model import SpeechRestFace,CURRENT
DEST=APP/'.local/speech-rest-v2'

STEPS=1600

def main():
    torch.set_num_threads(4);torch.manual_seed(103);rng=np.random.default_rng(7103)
    if (DEST/'selection.json').exists():raise ValueError('Completed speech-rest training exists')
    ds={}
    for name in ['train','dev','regression','quiet']:
        with np.load(DEST/(name+'.npz')) as z:ds[name]={k:z[k].copy() for k in z.files}
    with np.load(DEST/'geometry.npz') as z:geo={k:torch.from_numpy(z[k]) for k in z.files}
    net=SpeechRestFace();net.train()
    def inputs(d,ix):
        return torch.from_numpy(d['hidden'][ix]),torch.from_numpy(d['base'][ix]),torch.from_numpy(d['emotion'][ix,None,:]).expand(-1,60,-1)
    for name,d in ds.items():
        cache=DEST/(name+'-base.npy')
        if cache.exists():d['base']=np.load(cache)
        else:
            values=[]
            with torch.inference_mode():
                for start in range(0,len(d['hidden']),32):
                    ix=slice(start,start+32);h=torch.from_numpy(d['hidden'][ix]);b=torch.from_numpy(d['baseline'][ix]);e=torch.from_numpy(d['emotion'][ix,None,:]).expand(-1,60,-1);s=torch.zeros(len(h),60,3);s[:,:,1]=1
                    values.append(net.base(h,b,e,s).numpy())
            d['base']=np.concatenate(values);np.save(cache,d['base'])
        n=len(d['hidden']);following=np.arange(n)
        if name in ['train','dev']:
            for i in range(n-11):
                if d['cases'][i]==d['cases'][i+11]:following[i]=i+11
        elif name=='regression':following=np.minimum(np.arange(n)+1,n-1)
        d['following']=following;overlap=np.concatenate([d['valid'][:,25:30],d['valid'][following,:5]],axis=1);overlap[following==np.arange(n)]=False;d['overlap']=overlap
        print('cached',name,len(d['hidden']),flush=True)
    def masked(error,valid):return (error.square()*valid[:,:,None]).sum()/(valid.sum().clamp_min(1)*error.shape[-1])
    def loss_for(d,ix,p):
        p=p[:,15:45];base=torch.from_numpy(d['base'][ix,15:45]);valid=torch.from_numpy(d['valid'][ix]);vel=valid[:,1:]&valid[:,:-1]
        predicted_gap=p@geo['gap']+geo['rest'];target_gap=torch.from_numpy(d['target_gaps'][ix]);raw=torch.from_numpy(d['raw_gaps'][ix]);base_gap=base@geo['gap']+geo['rest']
        loss=32*masked(predicted_gap-target_gap,valid)
        # Penalize additional raw-teacher gap error rather than pinning every
        # old per-frame gap, which can force compensating jaw motion.
        regression=(torch.relu((predicted_gap-raw).square().mean(-1)-(base_gap-raw).square().mean(-1))*valid).sum()/valid.sum().clamp_min(1)
        loss=loss+64*regression
        for name,weight in [('corner',6.),('chin',3.),('mouth',1.)]:
            error=p@geo[name]-torch.from_numpy(d['target_'+name][ix]);loss=loss+weight*masked(error,valid)
            if name in ['chin','mouth']:loss=loss+.5*masked((error[:,1:]-error[:,:-1])/.1,vel)
        upper=(p-base)@geo['upper'];loss=loss+.2*masked(upper/.02,valid)
        loss=loss+.03*masked((p[:,:,14:41]-base[:,:,14:41])/.1,valid)
        return loss
    train=ds['train'];groups=[np.flatnonzero(train['voices']==v) for v in sorted(set(train['voices']))]
    def quiet_motion(d,p):
        valid=torch.from_numpy(d['valid']);p=p[:,15:45];pieces=[]
        for name in ['chin','mouth']:
            x=p@geo[name];steps=x[:,1:]-x[:,:-1];wrap=x[:,-1:]-x[:,:1];pieces.append(torch.cat([steps,wrap],dim=1).square().mean())
        return sum(pieces)/len(pieces)
    baseline_quiet=float(quiet_motion(ds['quiet'],torch.from_numpy(ds['quiet']['base'])))
    with np.load(APP/'.local/corner-appointment-regression/features.npz') as z:stress_raw_corner=z['corner_target']*10
    stress=ds['regression'];sv=stress['valid'];base_corner=(torch.from_numpy(stress['base'][:,15:45])@geo['corner']).numpy()[sv]
    stress_limit=float(np.sqrt(np.mean((base_corner[-48:]-stress_raw_corner[sv][-48:])**2)))*1.05
    @torch.inference_mode()
    def evaluate():
        d=ds['dev'];score=0.;n=0;gap_stats={};corner_stats={};tracks={};base_tracks={}
        for start in range(0,len(d['hidden']),64):
            ix=np.arange(start,min(start+64,len(d['hidden'])));h,b,e=inputs(d,ix);p=net.correct(h,b,e);value=loss_for(d,ix,p);score+=float(value)*len(ix);n+=len(ix)
            gp=(p[:,15:45]@geo['gap']+geo['rest']).numpy();gb=(b[:,15:45]@geo['gap']+geo['rest']).numpy();cp=(p[:,15:45]@geo['corner']).numpy();cb=(b[:,15:45]@geo['corner']).numpy();ch=(p[:,15:45]@geo['chin']).numpy();bh=(b[:,15:45]@geo['chin']).numpy()
            for j,k in enumerate(ix):
                control='neutral' if d['emotion'][k].sum()==0 else str(int(d['emotion'][k].argmax()));v=d['valid'][k];g=gap_stats.setdefault(control,[0.,0.]);g[0]+=float(np.square(gp[j,v]-d['raw_gaps'][k,v]).sum());g[1]+=float(np.square(gb[j,v]-d['raw_gaps'][k,v]).sum())
                c=corner_stats.setdefault(control,[0.,0.]);c[0]+=float(np.square(cp[j,v]-d['target_corner'][k,v]).sum());c[1]+=float(np.square(cb[j,v]-d['target_corner'][k,v]).sum())
                case=str(d['cases'][k]);tracks.setdefault(case,[]).append(ch[j,v]);base_tracks.setdefault(case,[]).append(bh[j,v])
        ratios={k:float(np.sqrt(a/max(b,1e-12))) for k,(a,b) in gap_stats.items()};corners={k:float(np.sqrt(a/max(b,1e-12))) for k,(a,b) in corner_stats.items()}
        tail_new=[];tail_base=[]
        for key,parts in tracks.items():
            tail_new.extend(np.diff(np.concatenate(parts)[-13:],axis=0).ravel()**2);tail_base.extend(np.diff(np.concatenate(base_tracks[key])[-13:],axis=0).ravel()**2)
        tail_ratio=float(np.sqrt(np.mean(tail_new)/max(np.mean(tail_base),1e-12)))
        q=ds['quiet'];qh,qb,qe=inputs(q,np.arange(len(q['hidden'])));quiet_ratio=float(quiet_motion(q,net.correct(qh,qb,qe)))/max(baseline_quiet,1e-12)
        r=ds['regression'];rh,rb,re=inputs(r,np.arange(len(r['hidden'])));rp=net.correct(rh,rb,re);corner=(rp[:,15:45]@geo['corner']).numpy()[sv];stress_rms=float(np.sqrt(np.mean((corner[-48:]-stress_raw_corner[sv][-48:])**2)))
        eligible=max(ratios.values())<=1.05 and max(corners.values())<=1.05 and stress_rms<=stress_limit and quiet_ratio<=.4 and tail_ratio<=.75
        return score/n,dict(gap_ratios=ratios,corner_ratios=corners,quiet_motion_variance_ratio=quiet_ratio,dev_last200_chin_step_ratio=tail_ratio,appointment_anger_corner_mm=stress_rms,eligible=eligible)
    freeze(DEST/'training-protocol.json',dict(steps=STEPS,seed=103,parent_sha256=sha(CURRENT/'model.onnx'),
        sources={n:sha(Path(__file__).with_name(n)) for n in ['train_speech_rest.py','speech_rest_model.py','prepare_speech_rest.py']},
        data={n:sha(DEST/(n+'.npz')) for n in ds},
        method='Learned 27-mouth residual, replicate-padded temporal decoder, frozen original 25 channels; no runtime silence detection or clip-end gate',
        target='Teacher geometry plus stationary teacher-derived quiet rest; 150ms release then 50ms training-label settle',
        acceptance='Raw-teacher gap RMS within5% and corner within5% on dev controls, existing anger fix within5%, quiet motion variance <=40%, dev final200ms chin steps <=75%',
        stress_limit_mm=stress_limit,labels='Appointment/anger explicit development sample, raw corner targets retained; not independent testing',losses='32 gap fit +64 per-frame raw gap nonregression;6 corners;3 chin;1 mouth; derivative geometry;quiet stationarity;stress retention'))
    params=[p for p in net.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=.0003,weight_decay=.001)
    best=float('inf');state=None;history=[];begun=time.monotonic()
    for step in range(1,STEPS+1):
        ix=np.concatenate([rng.choice(g,8,replace=True) for g in groups]);h,b,e=inputs(train,ix);p=net.correct(h,b,e)
        ni=train['following'][ix];nh,nb,ne=inputs(train,ni);following=net.correct(nh,nb,ne);ov=torch.from_numpy(train['overlap'][ix])
        continuity=masked(((p[:,40:50]-following[:,10:20])@geo['chin'])/.1,ov)+masked(((p[:,40:50]-following[:,10:20])@geo['mouth'])/.1,ov)
        q=ds['quiet'];qi=rng.integers(len(q['hidden']),size=8);qh,qb,qe=inputs(q,qi);qp=net.correct(qh,qb,qe)
        quiet_loss=loss_for(q,qi,qp)+4*quiet_motion(q,qp)/.01
        r=ds['regression'];ri=np.concatenate([rng.integers(len(r['hidden']),size=2),rng.choice(np.arange(max(0,len(r['hidden'])-3),len(r['hidden'])),2)])
        rh,rb,re=inputs(r,ri);rp=net.correct(rh,rb,re)
        loss=loss_for(train,ix,p)+continuity+quiet_loss+.5*loss_for(r,ri,rp)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite speech-rest loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
        optimizer.param_groups[0]['lr']=float(.0003*(.1+.45*(1+np.cos(np.pi*step/STEPS))))
        if step%100==0:
            score,checks=evaluate();eligible=checks['eligible'] and step>=300
            if eligible and score<best:best=score;best_step=step;state=copy.deepcopy(net.state_dict());torch.save(state,DEST/'best.pt')
            history.append(dict(step=step,score=score,**checks,seconds=time.monotonic()-begun));(DEST/'progress.json').write_text(json.dumps(history,indent=2))
            print('rest',step,round(score,3),'quiet',round(checks['quiet_motion_variance_ratio'],3),'tail',round(checks['dev_last200_chin_step_ratio'],3),'gap',round(max(checks['gap_ratios'].values()),3),'corner',round(checks['appointment_anger_corner_mm'],3),'eligible',eligible,'seconds',round(time.monotonic()-begun),flush=True)
    if state is None:raise ValueError('No rest checkpoint passed the declared gates')
    torch.save(state,DEST/'face.pt');freeze(DEST/'selection.json',dict(best_step=best_step,score=best,history=history));print('SELECTED',best_step,flush=True)

if __name__=='__main__':main()

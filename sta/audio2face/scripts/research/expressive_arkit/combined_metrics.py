"""Final multi-style coefficient scores, with a simple constant-pose control."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from gaze import NEW,STYLES
from train import load
from model import ExpressionAdapter
from gaze_model import GazeHead


def main():
    torch.set_num_threads(4)
    train,test=load('train'),load('test')
    with np.load(NEW/'train.npz') as z:train_eyes=z['eyes']
    with np.load(NEW/'test.npz') as z:test_eyes=z['eyes']
    state=torch.load(OUT/'adapter.pt',map_location='cpu',weights_only=True)
    adapter=ExpressionAdapter(train['protected'],np.zeros(96,np.float32),np.ones(96,np.float32))
    adapter.load_state_dict(state);adapter.eval()
    gaze=GazeHead(state['feature_mean'],state['feature_scale'])
    gaze.load_state_dict(torch.load(NEW/'gaze.pt',map_location='cpu',weights_only=True));gaze.eval()
    eye_ids=[1,2,3,4,8,9,10,11];mask=torch.ones(52);mask[eye_ids]=0
    face_templates=[];gaze_templates=[]
    for e in range(11):
        ix=np.arange(e,len(train['target']),11)
        residual=train['target'][ix]-train['baseline'][ix,15:45]
        face_templates.append(residual[train['valid'][ix]].mean(0))
        gaze_templates.append([train_eyes[ix,s][train['valid'][ix]].mean(0) for s in range(3)])
    stats={}
    for style in range(3):
        for start in range(0,len(test['hidden']),128):
            stop=min(start+128,len(test['hidden']));h=torch.from_numpy(test['hidden'][start:stop]);b=torch.from_numpy(test['baseline'][start:stop])
            e=torch.from_numpy(test['emotion'][start:stop,None,:]).expand(-1,60,-1);s=torch.zeros(len(h),60,3);s[:,:,style]=1
            with torch.inference_mode():
                prediction=(adapter(h,b,e)*mask+gaze.coefficients(gaze(h,e,s)))[:,15:45].numpy()
                target=test['target'][start:stop].copy()
                gt_eye=gaze.coefficients(torch.from_numpy(test_eyes[start:stop,style])).numpy()
                target[:,:,eye_ids]=gt_eye[:,:,eye_ids]
            for local,index in enumerate(range(start,stop)):
                condition=0 if not test['emotion'][index].any() else int(test['emotion'][index].argmax())+1
                template=np.clip(test['baseline'][index,15:45]+face_templates[condition],0,1)
                with torch.inference_mode():
                    gaze_pose=gaze.coefficients(torch.from_numpy(np.asarray(gaze_templates[condition][style]))).numpy()
                template[:,eye_ids]=gaze_pose[eye_ids]
                valid=test['valid'][index]
                error=prediction[local,valid]-target[local,valid]
                simple=template[valid]-target[local,valid]
                stat=stats.setdefault((str(test['voices'][index]),style),np.zeros(105,np.float64))
                stat[:52]+=np.square(error).sum(0);stat[52:104]+=np.square(simple).sum(0);stat[104]+=valid.sum()
    learned=np.mean([np.sqrt(s[:52]/s[104]) for s in stats.values()],axis=0)
    constant=np.mean([np.sqrt(s[52:104]/s[104]) for s in stats.values()],axis=0)
    report=dict(learned_coefficient_rms=float(learned.mean()),constant_pose_coefficient_rms=float(constant.mean()),
        relative_reduction=1-float(learned.mean()/constant.mean()),learned_channels=learned.tolist(),constant_channels=constant.tolist(),
        metric='Per-channel RMS, then equal mean over held-out voices and gaze styles. Targets retain paired expression differences on the protected articulation channels.',
        limitation='Known speech, single-seed small pilot; numerical fit comparison, not visual-quality or release confirmation.')
    freeze(NEW/'combined-metrics.json',report)
    names=json.loads((OUT/'channels.json').read_text())
    fig,ax=plt.subplots(figsize=(13,4.5));ix=np.arange(52)
    ax.bar(ix-.2,constant,.4,label='Constant expression/style pose');ax.bar(ix+.2,learned,.4,label='Audio-conditioned candidate')
    ax.set_xticks(ix,names,rotation=90,fontsize=7);ax.set_ylabel('Coefficient RMS');ax.legend();fig.tight_layout()
    fig.savefig(NEW/'channel-errors.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ['learned_channels','constant_channels']},indent=2))


if __name__=='__main__':main()

"""Standalone experimental plots and rendered corruption QA, not model editing."""
from common import *
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def main():
 s=json.loads((OUT/'summary.json').read_text());c=json.loads((OUT/'confirmation/summary.json').read_text())['fixed-confirmation'];keys=['shift-m8','shift-m4','shift-p4','shift-p8','underopen-50','missing-closure'];labels=['Delay320ms','Delay160ms','Advance160ms','Advance320ms','Reduced opening','Missing closure'];fig,ax=plt.subplots(figsize=(10,4),layout='constrained');x=np.arange(len(keys));w=.25
 for j,(name,r) in enumerate([('AVSR paper loss',s['av']['paper-av']),('SyncNet first set',s['sync']['calibrated']),('SyncNet confirmation',c)]):ax.bar(x+(j-1)*w,[r['comparisons'][k]['wins'] for k in keys],w,label=name)
 ax.axhline(6,color='black',linestyle='--',label='Predeclared6/8 minimum');ax.set(xticks=x,xticklabels=labels,ylim=(0,8.8),ylabel='Clips where correct animation scores better (out of8)',title='Frozen expert compatibility on James');ax.tick_params(axis='x',labelsize=9);ax.legend(fontsize=8,ncol=2);fig.savefig(OUT/'compatibility.png',dpi=160);plt.close(fig)
 case=json.loads((OUT/'cases.json').read_text())[4];n=case['frames'];rig=np.load(WORK/'arkit-student/rig-metrics.npz');a=np.fromfile(OUT/'tracks'/(case['id']+'-correct.f32'),'<f4').reshape(n,52);g=np.einsum('tc,ck->tk',a,rig['gap_basis'],optimize=False)+rig['rest']+rig['gap_offset'];b=np.fromfile(OUT/'tracks'/(case['id']+'-missing-closure.f32'),'<f4').reshape(n,52);ids=[int(np.median(g,-1).argmax()),int(np.abs(a-b).sum(-1).argmax())];fig,axes=plt.subplots(2,3,figsize=(8,5),layout='constrained')
 for row,t in enumerate(ids):
  for col,kind in enumerate(['correct','underopen-50','missing-closure']):
   frame=cv2.cvtColor(cv2.imread(str(OUT/'frames'/case['id']/kind/'mouth'/f'{t:04}.png')),cv2.COLOR_BGR2RGB);axes[row,col].imshow(frame);axes[row,col].axis('off');axes[row,col].set_title(kind if row==0 else f'frame{t}',fontsize=10)
 fig.suptitle('James corruption QA: opening peak / targeted closure');fig.savefig(OUT/'corruption-qa.png',dpi=140)
if __name__=='__main__':main()

"""Standalone scientific figures from the frozen diagnostic results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from common import *
from summarize import aggregate

def main():
 result=json.loads((OUT/'summary.json').read_text());curves=result['curves'];time=np.arange(225)/60;fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 groups=[('Warm speech histories',lambda r:r['kind']=='history' and r['a']!='cold' and r['b']!='cold'),('Cold → warm silence',lambda r:r['kind']=='history' and r['a']=='cold' and r['b']=='silence'),('Noise seeds, cold history',lambda r:r['kind']=='noise')]
 for ax,metric,title in zip(axes,['gap_mm','surface_mm'],['Nine-gap RMS difference','Dense 3D mouth RMS difference']):
  for label,select in groups:
   values=np.array([r[metric] for r in curves if select(r)]);ax.semilogy(time,np.maximum(np.sqrt(np.mean(values**2,axis=0)),1e-6),label=label)
  ax.axvspan(0,.25,color='grey',alpha=.15,label='Prefix may be in current audio');ax.axhline(.1,color='black',ls=':',lw=.8);ax.set(xlabel='Seconds after common speech begins',ylabel='RMS difference (mm)',title=title,xlim=(0,3.75));ax.legend(fontsize=7)
 fig.suptitle('16 matched suffixes; identical current windows after 250 ms');fig.savefig(OUT/'history-noise-decay.png',dpi=160);plt.close(fig)
 bins=[[.25,.75],[.75,1.25],[1.25,2.25],[2.25,3.25]];cases=json.loads((OUT/'cases.json').read_text());voices=sorted({r['engine']+':'+r['voice'] for r in cases});matrix=[];engine_rows={}
 for voice in voices:
  matrix.append([aggregate([r for r in result['comparisons'] if r['engine']+':'+r['voice']==voice and r['kind']=='history' and r['seed']==0 and r['a']!='cold' and r['b']!='cold' and r['bin']==b])['gap_rms_mm'] for b in bins])
 for engine in ['phonon','supertonic']:
  engine_rows[engine]={f'{b[0]}-{b[1]}':aggregate([r for r in result['comparisons'] if r['engine']==engine and r['kind']=='history' and r['seed']==0 and r['a']!='cold' and r['b']!='cold' and r['bin']==b]) for b in bins}
 fig,ax=plt.subplots(figsize=(8,5),layout='constrained');im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=max(map(max,matrix)),cmap='magma');ax.set(xticks=np.arange(4),xticklabels=['250–750 ms','750–1250 ms','1250–2250 ms','2250–3250 ms'],yticks=np.arange(8),yticklabels=voices,title='Fixed-noise warm-history gap differences (mm)')
 for y,row in enumerate(matrix):
  for x,v in enumerate(row):ax.text(x,y,f'{v:.3f}',ha='center',va='center',color='white' if v<max(map(max,matrix))*.6 else 'black',fontsize=8)
 fig.colorbar(im,ax=ax,label='Pooled gap RMS (mm)');fig.savefig(OUT/'history-by-voice.png',dpi=160);plt.close(fig)
 freeze(OUT/'engine-breakdown.json',engine_rows)
 print(json.dumps(engine_rows,indent=2))
if __name__=='__main__':main()

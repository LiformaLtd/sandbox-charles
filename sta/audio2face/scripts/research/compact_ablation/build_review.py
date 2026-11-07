from pathlib import Path
import argparse,json,hashlib,sys,wave,os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
APP=Path(__file__).resolve().parents[3]
H=Path(os.environ.get('A2F_ABLATION_OUTPUT',APP/'.local/compact-followup')).resolve()
p=argparse.ArgumentParser();p.add_argument('--provisional',action='store_true');p.add_argument('--folded',action='store_true');args=p.parse_args()
if args.provisional:chosen={str(w):f'w{w}-s37' for w in [96,128,192]}
else:chosen=json.loads((H/'study-summary.json').read_text())['selected_by_validation']
ROOT=H/'web/public/review';ROOT.mkdir(parents=True,exist_ok=True)
WORK=Path(os.environ['A2F_RESEARCH_WORK']).resolve()
models=[{'id':'teacher','title':'Teacher fitted to James'},{'id':'baseline','title':'Released v3 · 96 channels'}]+[{'id':str(w),'title':f'Longer training · {w} channels','name':chosen[str(w)]} for w in [96,128,192]]
if args.folded:
 seed=chosen['192'].split('-s')[1]
 models[-1]={'id':'folded','title':'Folded candidate · 96 channels','name':f'folded-s{seed}'}
for model in models:
 if model['id']=='teacher':continue
 path=WORK/'arkit-student-v3/james-arkit-student-v3.onnx' if model['id']=='baseline' else H/(model['name']+'.onnx')
 model['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
rows=json.loads((H/'confirmation/manifest.json').read_text());cases=[]
sys.path.insert(0,str(Path(__file__).resolve().parent));import metrics
rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=rig['gap_basis'];rest=rig['rest']+rig['gap_offset']
for index in [0,8,16,24,32,40]:
 row=rows[index];stem=row['stem']
 with np.load(H/'confirmation/labels'/f'{stem}.npz') as z:teacher=z['weights'].copy();gaps=z['gaps'].copy()
 teacher.astype('<f4').tofile(ROOT/f'{stem}-teacher.f32')
 track_values={'teacher':teacher}
 for model in models[1:]:
  name='baseline' if model['id']=='baseline' else model['name']
  cache=H/('confirmation-predictions-'+name+'-'+model['sha256'][:12])
  values=np.load(cache/f'{stem}.npy');assert values.shape==teacher.shape
  values.astype('<f4').tofile(ROOT/f'{stem}-{model["id"]}.f32')
  track_values[model['id']]=values
 pcm=np.fromfile(row['audio'],'<f4')
 with wave.open(str(ROOT/f'{stem}.wav'),'wb') as wav:
  wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
  wav.writeframes((np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())
 events=list(metrics.associate_events([{**r,'phone':r['phone'].lower()} for r in row['phones']],metrics.runs(np.median(gaps,axis=-1)<=.1)))
 wanted={'closure_m':'m','closure_b':'b','closure_p':'p'}.get(row['source'])
 eligible=[(phone,a,b) for phone,(a,b) in events if a>=24 and (wanted is None or phone==wanted)]
 if not eligible:eligible=[(phone,a,b) for phone,(a,b) in events]
 assert eligible,stem
 eligible.sort(key=lambda event:event[1])
 phone,a,b=eligible[0];a,b=int(a),int(b);contact=(a+b-1)//2
 snapshots=[{'frame':max(0,a-4),'title':f'Before /{phone}/ · {max(0,a-4)/60:.3f} s'},
            {'frame':contact,'title':f'Contact /{phone}/ · {contact/60:.3f} s'},
            {'frame':min(len(teacher)-1,b+4),'title':f'After /{phone}/ · {min(len(teacher)-1,b+4)/60:.3f} s'}]
 fig,ax=plt.subplots(figsize=(12,2.7))
 fig.subplots_adjust(top=.73,bottom=.20,left=.07,right=.99)
 times=np.arange(len(teacher))/60
 ax.plot(times,np.median(gaps,axis=-1),color='black',linewidth=1.8,label='Raw teacher')
 colors={'teacher':'#999999','baseline':'#9a89b5','96':'#2b8cbe','128':'#5aae61','192':'#d95f02','folded':'#d95f02'}
 for model in models:
  gap=np.einsum('fc,cg->fg',track_values[model['id']],basis,optimize=False)+rest
  ax.plot(times,np.median(gap,axis=-1),color=colors[model['id']],linewidth=1.,linestyle=':' if model['id']=='teacher' else '-',label=model['title'])
 ax.axhline(.1,color='#b2182b',linestyle='--',linewidth=.8,label='Contact threshold')
 ax.axvline(contact/60,color='#555555',linestyle=':',linewidth=.8,label='Chosen contact frame')
 ax.set(xlabel='Time (s)',ylabel='Median lip gap (rig units)',xlim=(0,len(pcm)/16000))
 fig.legend(*ax.get_legend_handles_labels(),ncol=4,fontsize=7,loc='upper center',bbox_to_anchor=(.52,.99),frameon=False)
 ax.spines[['top','right']].set_visible(False)
 fig.savefig(ROOT/f'{stem}-gaps.png',dpi=160);plt.close(fig)
 cases.append({'stem':stem,'title':row['source'].replace('_',' ').title(),'text':row['text'],'frames':len(teacher),'snapshots':snapshots})
(ROOT/'manifest.json').write_text(json.dumps({'models':models,'cases':cases,'provisional':args.provisional},indent=2))
print('Built six predetermined matched review clips.', 'Provisional first-seed preview.' if args.provisional else 'Models selected by validation.')

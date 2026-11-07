from pathlib import Path
import os
import importlib.util,argparse,json,hashlib,time
import numpy as np
APP=Path(__file__).resolve().parents[3]
HERE=Path(os.environ.get('A2F_ABLATION_OUTPUT',APP/'.local/compact-ablation')).resolve()
HERE.mkdir(parents=True,exist_ok=True)
WORK=Path(os.environ.get('A2F_RESEARCH_WORK',APP/'.local/research-corpora')).resolve()
V3=WORK/'arkit-student-v3'
import evaluation_support as ev
def infer(session,audio):
    size=session.get_inputs()[0].shape[1];padding=(size-16000)//2
    first=15+padding*60//16000;offsets=np.arange(15,45)*16000//60
    pieces=[];timestamps=[]
    for original in range(-8000,len(audio)-4000,8000):
        valid=(original+offsets>=0)&(original+offsets<len(audio))
        if not valid.any():continue
        start=original-padding;x=np.zeros((1,size),np.float32)
        lo,hi=max(start,0),min(start+size,len(audio));x[0,lo-start:hi-start]=audio[lo:hi]
        pieces.append(session.run(None,{'pcm':x})[0][0,first:first+30][valid]);timestamps.extend((original+offsets)[valid])
    pred=np.concatenate(pieces)
    np.testing.assert_array_equal(timestamps,np.arange((len(audio)*60+15999)//16000)*16000//60)
    return pred
ev.metrics.infer=infer

def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('model',type=Path);args=p.parse_args()
    model=ev.session(args.model);cache=HERE/('predictions-'+args.name+'-'+hashlib.sha256(args.model.read_bytes()).hexdigest()[:12]);cache.mkdir(exist_ok=True)
    result={'sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'bytes':args.model.stat().st_size};perclip={}
    for corpus,root in [('new',V3),('prior_v2',WORK/'arkit-student-v2'),('original',WORK/'arkit-student')]:
        records=ev.collect(root,model,cache,'test')
        groups={corpus:records}
        if corpus=='new':
            groups.update({f'new_{source}':[r for r in records if r[0]['source']==source] for source in sorted({r[0]['source'] for r in records})})
            phones={r[0]['stem']:[{**p,'phone':p['phone'].lower()} for p in r[0].get('phones',[])] for r in records}
            result['phone_events'],_=ev.metrics.event_scores([(*r,r[3]) for r in records],3,phones)
        if corpus=='prior_v2':groups['prior_v2_natural']=[r for r in records if r[0].get('kind')=='natural']
        for group,rows in groups.items():
            result[group]=ev.summarize(rows)
            summaries=[]
            for row,truth,fit,pred in rows:
                a=np.median(truth,axis=-1)<=.1;b=np.median(pred,axis=-1)<=.1
                summaries.append({'stem':row['stem'],'squared_error':float(((truth-pred)**2).sum()),'values':truth.size,'tp':int((a&b).sum()),'actual':int(a.sum()),'closed':int(b.sum())})
            perclip[group]=summaries
    result['per_clip']=perclip
    (HERE/f'{args.name}-quality.json').write_text(json.dumps(result,indent=2))
    for group in ['new','new_subtitles','new_fresh','prior_v2','original']:print(args.name,group,result[group]['student'],flush=True)
if __name__=='__main__':main()

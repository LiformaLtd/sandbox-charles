from pathlib import Path
import importlib.util,os,sys,json,hashlib
import numpy as np
APP=Path(__file__).resolve().parents[3]
H=Path(os.environ.get('A2F_ABLATION_OUTPUT',APP/'.local/compact-followup')).resolve()
sys.path.insert(0,str(Path(__file__).resolve().parent))
spec=importlib.util.spec_from_file_location('eval',Path(__file__).with_name('evaluate.py'));ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
name=sys.argv[1]
path=ev.WORK/'arkit-student-v3/james-arkit-student-v3.onnx' if name=='baseline' else H/(name+'.onnx')
cache=H/('confirmation-predictions-'+name+'-'+hashlib.sha256(path.read_bytes()).hexdigest()[:12]);cache.mkdir(exist_ok=True)
model=ev.ev.session(path)
rows=ev.ev.collect(H/'confirmation',model,cache,'test')
result={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'all':ev.ev.summarize(rows)}
for source in sorted({r[0]['source'] for r in rows}):result[source]=ev.ev.summarize([r for r in rows if r[0]['source']==source])
phones={r[0]['stem']:[{**p,'phone':p['phone'].lower()} for p in r[0].get('phones',[])] for r in rows}
result['phone_events'],_=ev.ev.metrics.event_scores([(*r,r[3]) for r in rows],3,phones)
result['per_clip']=[]
for row,truth,fit,pred in rows:
 a=np.median(truth,axis=-1)<=.1;b=np.median(pred,axis=-1)<=.1
 result['per_clip'].append(dict(stem=row['stem'],source=row['source'],squared_error=float(((truth-pred)**2).sum()),values=truth.size,tp=int((a&b).sum()),actual=int(a.sum()),closed=int(b.sum())))
(H/('confirmation-'+name+'.json')).write_text(json.dumps(result,indent=2))
print(name,result['all']['student'],flush=True)

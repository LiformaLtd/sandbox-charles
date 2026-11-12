"""Complete training-corpus fit, with exact playback crops and frozen held-out comparisons."""
from common import *
STUDY=OUT;OUT=OUT/'fit';SYNC=STUDY
MODELS={a:STUDY/a/'model.onnx' for a in ['control','protected']}
PARTS=[('old-train',SCALE/'datasets/old-train','train'),('new-train',SCALE/'datasets/new-train','train'),('replay-train',WORK/'arkit-student-v3','train'),('extra-train',WORK/'arkit-student-v3/extra','train'),('common-dev',SCALE/'datasets/common-dev','dev'),('replay-dev',WORK/'arkit-student-v3','dev')]
import numpy as np,onnx,onnxruntime as ort,mlx.core as mx
from onnx import numpy_helper
from mlx_backend import Student
from features import Dataset
from evaluate import evaluate_cohort,aggregate
import closure_metrics
sys.path.insert(0,str(APP/'scripts/research/closure_protected'))
from inference import infer
import metrics
metrics.infer=infer

def score_records(records):
 details=[]
 for row,truth,weights,pred,gaps in records:
  a,b=np.median(truth,-1)<=.1,np.median(gaps,-1)<=.1;opening=np.median(truth,-1)>.35
  details.append(dict(stem=row['stem'],text=row['text'],engine=row['engine'],voice=row['voice'],gender=row['gender'],kind=row['kind'],condition=row['condition'],source=row['source'],squared_gap_error=float(np.square(gaps-truth).sum()),gap_values=truth.size,teacher_contacts=int(a.sum()),student_contacts=int(b.sum()),contact_hits=int((a&b).sum()),opening_squared_error=float(np.square(gaps[opening]-truth[opening]).sum()),opening_values=int(opening.sum())*9))
 conditions={}
 for c in sorted({r['condition'] for r in details}):
  rows=[r for r in details if r['condition']==c];conditions[c]=dict(**aggregate(rows),engines={e:aggregate([r for r in rows if r['engine']==e]) for e in sorted({r['engine'] for r in rows})},closure_audit=closure_metrics.score([(r,t,g) for r,t,w,p,g in records if r['condition']==c],lead=.1)['summary'])
 return dict(conditions=conditions,clips=details)

def audit_cache(name,root,rows):
 meta=json.loads((SCALE/'features'/name/'complete.json').read_text());source=APP/'scripts/research/data_scale/features.py';assert sha(source)==meta['source_sha256'];fingerprint=hashlib.sha256((root/'manifest.json').read_bytes()+source.read_bytes())
 for row in rows:
  digest=sha(row['audio']);assert not row.get('sha256_audio') or digest==row['sha256_audio'];fingerprint.update(digest.encode());fingerprint.update((root/'labels'/(row['stem']+'.npz')).read_bytes())
 assert fingerprint.hexdigest()==meta['input_sha256'],name

def main():
 mx.set_default_device(mx.gpu);rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=rig['gap_basis'];rest=rig['rest']+rig['gap_offset'];active=rig['active']
 protocol=dict(models={n:dict(path=str(p),sha256=sha(p)) for n,p in MODELS.items()},sources={str(p.relative_to(APP)):sha(p) for p in [HERE/'measure_training.py',HERE/'common.py',APP/'scripts/research/data_scale/features.py',APP/'scripts/research/data_scale/mlx_backend.py',APP/'scripts/research/data_scale/evaluate.py',APP/'scripts/research/closure_protected/inference.py',APP/'scripts/research/multivoice/closure_metrics.py']},parts={n:dict(root=str(r),split=s,manifest_sha256=sha(r/'manifest.json')) for n,r,s in PARTS},scope='All primary and replay TRAINING frames, common/replay development separately, existing fixed fresh confirmation; no checkpoint selection or new training',parity_tolerance=5e-5)
 prior=json.loads((SCALE/'training-protocol.json').read_text())
 for name,digest in prior['array_sha256'].items():assert sha(SCALE/name)==digest,name
 protocol['array_sha256']=prior['array_sha256'];freeze(OUT/'protocol.json',protocol)
 datasets={n:Dataset([n]) for n,_,_ in PARTS};references={};reports={}
 for name,root,split in PARTS:audit_cache(name,root,[r for r in json.loads((root/'manifest.json').read_text()) if r['split']==split])
 for arm,path in MODELS.items():
  values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(path).graph.initializer};net=Student(values);cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True);opt=ort.SessionOptions();opt.intra_op_num_threads=2;session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider']);cohorts={};parity=[];allrows=[];normalized=[];fitstats={}
  for name,root,split in PARTS:
   dataset=datasets[name];data=dataset.parts[0];rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']==split];pred=np.empty((dataset.length,30,39),np.float32)
   for start in range(0,dataset.length,128):pred[start:start+128]=np.array(mx.clip(net(mx.array(data['mel'][start:start+128]))[:,15:45],0,1))
   records=[];offset=0;fixed={0,len(rows)-1};fixed.update(next(i for i,r in enumerate(rows) if r.get('engine','phonon')+':'+r.get('voice','Easton')==v) for v in sorted({r.get('engine','phonon')+':'+r.get('voice','Easton') for r in rows}));stats={};nr=[]
   for i,row in enumerate(rows):
    row=dict(row);row.setdefault('sha256_audio',sha(row['audio']));row.setdefault('engine','phonon');row.setdefault('voice','Easton');row.setdefault('gender','male');row.setdefault('kind','historical');row.setdefault('condition','original');row.setdefault('source',name);row['phones']=[dict(p,phone=p['phone'].lower()) for p in row['phones']]
    samples=Path(row['audio']).stat().st_size//4;n=len(range(-8000,samples-4000,8000));valid=data['valid'][offset:offset+n];clipped=pred[offset:offset+n][valid];track=np.zeros((len(clipped),52),np.float32);track[:,active]=clipped
    with np.load(root/'labels'/(row['stem']+'.npz')) as z:
     target=z['weights'];rawgap=z['gaps']
     if 'audio_sha256' in z:assert str(z['audio_sha256'])==row['sha256_audio']
    np.testing.assert_array_equal(data['weights'][offset:offset+n][valid],target[:,active]);np.testing.assert_array_equal(data['gaps'][offset:offset+n][valid],rawgap);assert len(track)==len(target)
    targetgap=np.einsum('tc,ck->tk',target,basis,optimize=False)+rest;estimate=np.einsum('tc,ck->tk',track,basis,optimize=False)+rest;key=(row['condition'],row['engine']+':'+row['voice']);v=stats.setdefault(key,np.zeros(8,np.float64));difference=(track[:,active]-target[:,active]);v += [np.square(estimate-targetgap).sum(),rawgap.size,np.square(targetgap-rawgap).sum(),np.square(difference).sum(),difference.size,np.square(difference/values['student.scale']).sum(),len(track),np.abs(difference).sum()]
    dest=cache/(row['sha256_audio']+'.npy')
    if dest.exists():np.testing.assert_array_equal(np.load(dest),track)
    else:np.save(dest,track)
    if i in fixed:
     expected=infer(session,np.fromfile(row['audio'],'<f4'));delta=float(np.abs(expected-track).max());assert delta<5e-5,(arm,name,i,delta);parity.append(dict(part=name,stem=row['stem'],delta=delta))
    offset+=n;nr.append(row);records.append((row,rawgap,target,track,estimate))
   assert offset==dataset.length
   fitstats[name]=[dict(condition=k[0],voice=k[1],fit_gap_rms_mm=float(10*np.sqrt(v[0]/v[1])),reference_gap_rms_mm=float(10*np.sqrt(v[2]/v[1])),coefficient_rms=float(np.sqrt(v[3]/v[4])),normalized_coefficient_rms=float(np.sqrt(v[5]/v[4])),coefficient_mae=float(v[7]/v[4]),frames=int(v[6]),sums=v.tolist()) for k,v in stats.items()]
   cohorts[name]=score_records(records);normalized+=nr
   if name in ['old-train','new-train']:allrows+=cohorts[name]['clips']
   print(arm,name,cohorts[name]['conditions']['original']['macro_gap_rmse_mm'],flush=True)
  from evaluate import aggregate
  pooled=aggregate([r for r in allrows if r['condition']=='original']);prior_eval=json.loads((SYNC/f'evaluation-{arm}.json').read_text());report=dict(model_sha256=sha(path),cohorts=cohorts,primary_original=pooled,fit_statistics=fitstats,native_parity=parity,fresh=prior_eval['fresh']['conditions'],fresh_source_sha256=sha(SYNC/f'evaluation-{arm}.json'))
  freeze(OUT/f'evaluation-{arm}.json',report);reports[arm]=report
 freeze(OUT/'complete.json',dict(models={a:r['model_sha256'] for a,r in reports.items()},protocol_sha256=sha(OUT/'protocol.json')))
if __name__=='__main__':main()

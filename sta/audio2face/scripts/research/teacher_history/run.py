"""Full history×noise factorial plus deterministic/reset/stateless controls; no training."""
import time,numpy as np,onnxruntime as ort
from common import *
from projector import backend,verify,animate
from app.audio2face_v3 import STATE_SHAPE
sys.path.insert(0,str(APP/'scripts/research/compact_ablation'))
from metrics import infer

def track(case,history,seed,run,protocol_sha,tag=None,reset=False):
 name=tag or history+'-s'+str(seed);root=OUT/'tracks'/case['id'];root.mkdir(parents=True,exist_ok=True);path=root/(name+'.npz')
 if path.exists():
  with np.load(path) as z:
   assert str(z['protocol_sha256'])==protocol_sha
   return {k:z[k].copy() for k in ['surface','gaps','states']}|{'trace':json.loads(str(z['trace_json']))}
 item=case['audio'][history];assert sha(item['path'])==item['sha256'];pcm=np.fromfile(item['path'],'<f4');prefix=item['prefix_samples'];calls=0;window_hashes={};states=[];noise_id=None;noise_sha=None
 def wrapped(inputs):
  nonlocal calls,noise_id,noise_sha
  start=-16000+calls*8000;calls+=1
  if noise_id is None:noise_id=id(inputs['noise']);noise_sha=array_sha(inputs['noise'])
  assert id(inputs['noise'])==noise_id
  effective=dict(inputs,input_latents=np.zeros(STATE_SHAPE,np.float32)) if reset else inputs
  if prefix<=start<=prefix+48000:
   window_hashes[str(start-prefix)]=array_sha(inputs['window']);states.append(effective['input_latents'].copy())
  return run(effective)
 indices=[];blocks=[]
 for ix,values in animate(pcm,wrapped,identity='James',seed=seed):indices.append(ix);blocks.append(values)
 times=np.concatenate(indices);np.testing.assert_array_equal(times,np.arange(len(times))*16000//60)
 values=np.concatenate(blocks);first=prefix*60//16000;values=values[first:first+225];assert values.shape==(225,6810) and np.isfinite(values).all()
 trace=dict(case=case['id'],history=history,seed=seed,reset=reset,audio_sha256=item['sha256'],noise_sha256=noise_sha,common_window_sha256=window_hashes,calls=calls)
 surface=values[:,:6801].reshape(225,2267,3).copy();gaps=values[:,6801:].copy();states=np.stack(states);assert len(states)==7
 temp=path.with_suffix('.part.npz');np.savez_compressed(temp,surface=surface,gaps=gaps,states=states,trace_json=json.dumps(trace),protocol_sha256=protocol_sha);temp.replace(path)
 return dict(surface=surface,gaps=gaps,states=states,trace=trace)

def main():
 protocol=json.loads((OUT/'protocol.json').read_text());assert sha(BASELINE)==EXPECTED
 for name,digest in protocol['source_sha256'].items():assert sha(APP/name)==digest,name
 cases=json.loads((OUT/'cases.json').read_text());protocol_sha=sha(OUT/'protocol.json')
 freeze(OUT/'projection-parity.json',verify(cases[0]));run=backend();options=ort.SessionOptions();options.intra_op_num_threads=2;student=ort.InferenceSession(str(BASELINE),sess_options=options,providers=['CPUExecutionProvider']);began=time.monotonic();checks=[];noise_hashes={}
 for case in cases:
  ref=None
  for seed in protocol['seeds']:
   for history in protocol['histories']:
    row=track(case,history,seed,run,protocol_sha)
    if ref is None:ref=row
    assert row['trace']['common_window_sha256']==ref['trace']['common_window_sha256']
    if seed in noise_hashes:assert row['trace']['noise_sha256']==noise_hashes[seed]
    else:noise_hashes[seed]=row['trace']['noise_sha256']
  repeated=track(case,'cold',0,run,protocol_sha,tag='repeat-cold-s0')
  np.testing.assert_array_equal(repeated['surface'],ref['surface']);np.testing.assert_array_equal(repeated['gaps'],ref['gaps'])
  reset_cold=track(case,'cold',0,run,protocol_sha,tag='reset-cold-s0',reset=True);reset_speech=track(case,'speech-a',0,run,protocol_sha,tag='reset-speech-a-s0',reset=True)
  np.testing.assert_array_equal(reset_cold['surface'][15:],reset_speech['surface'][15:]);np.testing.assert_array_equal(reset_cold['gaps'][15:],reset_speech['gaps'][15:])
  predictions={}
  for history in protocol['histories']:
   item=case['audio'][history];prediction=infer(student,np.fromfile(item['path'],'<f4'));first=item['prefix_samples']*60//16000;predictions[history]=prediction[first:first+225]
   if history!='cold':np.testing.assert_array_equal(predictions['cold'][15:],predictions[history][15:])
  np.save(OUT/'tracks'/case['id']/'student.npy',predictions['cold'])
  checks.append(dict(case=case['id'],windows_identical=True,repeat_identical=True,reset_identical=True,student_identical=True));freeze(OUT/'checks-progress.json',checks) if not (OUT/'checks-progress.json').exists() else (OUT/'checks-progress.json').write_text(json.dumps(checks,indent=2))
  print(case['id'],case['engine'],case['voice'],'factorial+controls passed; seconds',round(time.monotonic()-began),flush=True)
 files={str(p.relative_to(OUT)):sha(p) for p in (OUT/'tracks').rglob('*') if p.is_file()}
 freeze(OUT/'completion.json',dict(cases=len(cases),checks=checks,noise_sha256=noise_hashes,files_sha256=files,source_sha256=sha(__file__),seconds=time.monotonic()-began));print('History/noise diagnostic complete.',flush=True)
if __name__=='__main__':main()

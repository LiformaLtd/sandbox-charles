from common import *
from inference import infer
import numpy as np,onnxruntime as ort

def main():
 rows=json.loads((OUT/'confirmation/variants/manifest.json').read_text());fixtures=[]
 for engine in ['phonon','supertonic']:
  for condition in ['original','half','boost','quarter']:
   row=next(r for r in rows if r['engine']==engine and r['condition']==condition);fixtures.append((engine+'-'+condition,Path(row['audio']).read_bytes()))
  name='phonon-browser.f32' if engine=='phonon' else 'supertonic.f32';fixtures.append((engine+'-packed',(APP/'.local/packed-diagnostic'/name).read_bytes()))
 fixtures.append(('short-tail',fixtures[0][1][:1800*4]));root=OUT/'browser/parity';root.mkdir(parents=True,exist_ok=True);cases=[]
 for name,audio in fixtures:freeze(root/(name+'.f32'),audio);cases.append(dict(name=name,audio=name+'.f32',reference={}))
 models={'v4':APP/'.local/compact-feature-v1/control/model.onnx',**{a:OUT/a/'model.onnx' for a in ['control','protected']}}
 for arm,path in models.items():
  options=ort.SessionOptions();options.intra_op_num_threads=2;s=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
  for c in cases:
   pred=infer(s,np.fromfile(root/c['audio'],'<f4'));filename=c['name']+'-'+arm+'-cpu.f32';freeze(root/filename,pred.astype('<f4').tobytes());c['reference'][arm]=filename
 freeze(root/'parity-cases.json',cases);freeze(root/'fixture-lock.json',dict(model_sha256={a:sha(p) for a,p in models.items()},source_sha256=sha(__file__),audio_sha256={c['audio']:sha(root/c['audio']) for c in cases}))
if __name__=='__main__':main()

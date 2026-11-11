"""Cache fixed FFT power, not log-mel: gradients must pass through the mel projection."""
from common import *
import numpy as np,onnx
from onnx import numpy_helper
from features import Dataset as OriginalDataset

ROOTS={'old-train':(SCALE/'datasets/old-train','train'),'new-train':(SCALE/'datasets/new-train','train'),
       'common-dev':(SCALE/'datasets/common-dev','dev'),'replay-train':(WORK/'arkit-student-v3','train'),
       'extra-train':(WORK/'arkit-student-v3/extra','train'),'replay-dev':(WORK/'arkit-student-v3','dev')}

def build(name,window,bank):
 source,split=ROOTS[name];dest=POWER/name;dest.mkdir(parents=True,exist_ok=True)
 complete=dest/'complete.json'
 if complete.exists():
  meta=json.loads(complete.read_text())
  if sha(dest/'power.npy')!=meta['power_sha256']:raise ValueError('Changed power cache')
  return
 old=OriginalDataset([name]);rows=[r for r in json.loads((source/'manifest.json').read_text()) if r['split']==split]
 path=dest/'power.part.npy';power=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(old.length,257,101))
 pending=[];indices=[];cursor=0;max_delta=0;audio_hashes={}
 def flush():
  nonlocal max_delta
  if not pending:return
  pcm=np.stack(pending);windows=np.lib.stride_tricks.sliding_window_view(np.pad(pcm,((0,0),(200,200))),400,axis=1)[:,::160]
  fft=np.fft.rfft(windows*window,n=512,axis=-1)
  values=(fft.real**2+fft.imag**2).astype(np.float32).transpose(0,2,1)
  if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid spectral power')
  power[indices]=values
  reproduced=np.log1p(np.einsum('bft,mf->bmt',values,bank,optimize=False)*10)
  expected=old.batch(indices)['mel'];max_delta=max(max_delta,float(np.abs(reproduced-expected).max()))
  if max_delta>2e-5:raise ValueError('Cached waveform/frontend alignment mismatch: '+str(max_delta))
  pending.clear();indices.clear()
 for row in rows:
  digest=sha(row['audio']);audio_hashes[str(row['audio'])]=digest
  if row.get('sha256_audio') and row['sha256_audio']!=digest:raise ValueError('Source audio changed')
  pcm=np.fromfile(row['audio'],'<f4')
  for start in range(-8000,len(pcm)-4000,8000):
   if cursor>=old.length:raise ValueError('Extra waveform window')
   x=np.zeros(16000,np.float32);lo,hi=max(start,0),min(start+16000,len(pcm));x[lo-start:hi-start]=pcm[lo:hi]
   valid=(start+np.arange(15,45)*16000//60>=0)&(start+np.arange(15,45)*16000//60<len(pcm))
   if not np.array_equal(valid,old.batch([cursor])['valid'][0]):raise ValueError('Valid crop mismatch')
   pending.append(x);indices.append(cursor);cursor+=1
   if len(pending)==128:flush()
 flush()
 if cursor!=old.length:raise ValueError('Missing waveform windows')
 power.flush();del power;path.replace(dest/'power.npy')
 write(complete,dict(windows=cursor,power_sha256=sha(dest/'power.npy'),manifest_sha256=sha(source/'manifest.json'),audio_sha256=audio_hashes,max_reconstructed_logmel_error=max_delta,source_sha256=sha(__file__)))
 print(name,cursor,'windows, maximum logmel delta',max_delta,flush=True)

class Dataset:
 def __init__(self,names):
  self.original=OriginalDataset(names);self.length=self.original.length;self.voices=self.original.voices
  self.parts=[];self.starts=[];start=0
  for name in names:
   root=POWER/name;meta=json.loads((root/'complete.json').read_text());part=np.load(root/'power.npy',mmap_mode='r')
   if len(part)!=meta['windows']:raise ValueError('Incomplete power cache')
   self.parts.append(part);self.starts.append(start);start+=len(part)
  if start!=self.length:raise ValueError('Data sources differ')
 def batch(self,indices):
  indices=np.asarray(indices);result=self.original.batch(indices)
  del result['mel'];result['power']=np.empty((len(indices),257,101),np.float32)
  for start,part in zip(self.starts,self.parts):
   mask=(indices>=start)&(indices<start+len(part))
   if mask.any():result['power'][mask]=part[indices[mask]-start]
  return result

def main():
 if sha(BASELINE)!=EXPECTED:raise ValueError('Unexpected baseline')
 values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer}
 for name in ROOTS:build(name,values['filters'][0,0],values['mel'][:,:,0])
if __name__=='__main__':main()

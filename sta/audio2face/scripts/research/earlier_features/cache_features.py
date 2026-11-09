"""Earlier features and one identical fixed prediction cache for both arms."""
from common import *
import numpy as np,onnx,mlx.core as mx
from onnx import numpy_helper
from features import Dataset
from residual_model import Student,gelu
PARTS=['old-train','new-train','replay-train','extra-train','common-dev','replay-dev']

class FrozenDataset(Dataset):
    def __init__(self,names,arm):
        super().__init__(names)
        for name,part in zip(names,self.parts):
            path=OUT/'features'/name/'early.npy' if arm=='early' else PREVIOUS/'features'/name/'hidden.npy'
            part['mel']=np.load(path,mmap_mode='r');part['base_p']=np.load(OUT/'features'/name/'base-p.npy',mmap_mode='r')
    def batch(self,indices):
        b=super().batch(indices);p=np.empty((len(indices),60,39),np.float32);indices=np.asarray(indices)
        for start,part in zip(self.starts,self.parts):
            mask=(indices>=start)&(indices<start+len(part['mel']))
            if mask.any():p[mask]=part['base_p'][indices[mask]-start]
        b['mel']=np.concatenate([b['mel'],p],axis=-1);return b

def main():
    assert sha(BASELINE)=='ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da'
    prior=json.loads((SCALE/'training-protocol.json').read_text())
    for name,digest in prior['array_sha256'].items():assert sha(SCALE/name)==digest,name
    values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Student(values);net.freeze();mx.set_default_device(mx.gpu)
    freeze(OUT/'features/protocol.json',dict(source_sha256=sha(__file__),baseline_sha256=sha(BASELINE),array_sha256=prior['array_sha256']))
    for name in PARTS:
        data=Dataset([name]);late_path=PREVIOUS/'features'/name/'hidden.npy';old=json.loads((PREVIOUS/'features'/name/'complete.json').read_text());assert old['sha256']==sha(late_path)
        late=np.load(late_path,mmap_mode='r');assert late.shape==(data.length,60,96);dest=OUT/'features'/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():
            meta=json.loads((dest/'complete.json').read_text());assert meta['early_sha256']==sha(dest/'early.npy') and meta['prediction_sha256']==sha(dest/'base-p.npy');continue
        early=np.lib.format.open_memmap(dest/'early.part.npy',mode='w+',dtype=np.float32,shape=late.shape);prediction=np.lib.format.open_memmap(dest/'base-p.part.npy',mode='w+',dtype=np.float32,shape=(data.length,60,39));error=0.
        for start in range(0,data.length,128):
            mel=mx.array(data.parts[0]['mel'][start:start+128]);e=gelu(net.encoder(mel.transpose(0,2,1)));lo=mx.take(e,net.left,axis=1);hi=mx.take(e,net.left+1,axis=1);e=lo+net.fraction*(hi-lo)
            p=mx.array(late[start:start+128])@mx.array(values['student.output.weight'][:,:,0]).T+mx.array(values['student.output.bias']);early[start:start+128]=np.array(e);prediction[start:start+128]=np.array(p)
            if start==0:error=float(mx.max(mx.abs(net(mel)-(p*net.scale+net.mean))))
        assert error<5e-5,error
        early.flush();prediction.flush();del early,prediction
        (dest/'early.part.npy').replace(dest/'early.npy');(dest/'base-p.part.npy').replace(dest/'base-p.npy')
        freeze(dest/'complete.json',dict(windows=data.length,early_sha256=sha(dest/'early.npy'),prediction_sha256=sha(dest/'base-p.npy'),late_sha256=sha(late_path),base_parity=error));print(name,data.length,'early/prediction cached',flush=True)

if __name__=='__main__':main()

"""Once-only float32 frozen-v4 audio features, shared by both training arms."""
from common import *
import numpy as np,onnx,mlx.core as mx
from onnx import numpy_helper
from features import Dataset
from residual_model import Student,features

PARTS=['old-train','new-train','replay-train','extra-train','common-dev','replay-dev']

class FrozenDataset(Dataset):
    def __init__(self,names):
        super().__init__(names)
        for name,part in zip(names,self.parts):part['mel']=np.load(OUT/'features'/name/'hidden.npy',mmap_mode='r')

def main():
    assert sha(BASELINE)=='ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da'
    prior=json.loads((SCALE/'training-protocol.json').read_text())
    for name,digest in prior['array_sha256'].items():assert sha(SCALE/name)==digest,name
    values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Student(values);net.freeze();mx.set_default_device(mx.gpu)
    lock=dict(source_sha256=sha(__file__),model_source_sha256=sha(HERE/'residual_model.py'),baseline_sha256=sha(BASELINE),array_sha256=prior['array_sha256'])
    freeze(OUT/'features/protocol.json',lock)
    for part in PARTS:
        data=Dataset([part]);dest=OUT/'features'/part;dest.mkdir(parents=True,exist_ok=True);source=SCALE/'features'/part/'complete.json'
        if (dest/'complete.json').exists():assert json.loads((dest/'complete.json').read_text())['sha256']==sha(dest/'hidden.npy');continue
        hidden=np.lib.format.open_memmap(dest/'hidden.part.npy',mode='w+',dtype=np.float32,shape=(data.length,60,96));error=0.
        for start in range(0,data.length,128):
            x=mx.array(data.parts[0]['mel'][start:start+128]);h=features(net,x);hidden[start:start+128]=np.array(h)
            if start==0:error=float(np.max(np.abs(np.array(net(x))-(np.array(h)@values['student.output.weight'][:,:,0].T+values['student.output.bias'])*values['student.scale']-values['student.mean'])))
        hidden.flush();del hidden;(dest/'hidden.part.npy').replace(dest/'hidden.npy');assert error<5e-5,error
        freeze(dest/'complete.json',dict(windows=data.length,sha256=sha(dest/'hidden.npy'),source_metadata_sha256=sha(source),head_reconstruction_delta=error));print(part,data.length,'cached',flush=True)

if __name__=='__main__':main()

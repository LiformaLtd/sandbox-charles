"""Independent added channels, not Net2Wider replicas; training-only feature heads."""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
import numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_backend import gelu

class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.mix = nn.Conv1d(width,width,1)
    def __call__(self, x): return x + .5*self.mix(gelu(self.conv(x)))

class Predictor(nn.Module):
    def __init__(self, values, width=96):
        super().__init__()
        if width not in [96,192]: raise ValueError('Unsupported research width')
        self.encoder = nn.Conv1d(64,width,5,padding=2)
        self.blocks = [Block(width,d) for d in [1,2,4,8,16,1,2,4]]
        self.output = nn.Conv1d(width,39,1)
        self.mean = mx.array(values['student.mean'])
        self.scale = mx.array(values['student.scale'])
        self.left = mx.array(values['student.left'].astype(np.int32))
        self.fraction = mx.array(values['student.fraction'].transpose(0,2,1))
        weights = []
        for name, value in tree_flatten(self.parameters()):
            if name in ['mean','scale','left','fraction']: continue
            old = values['student.'+name]
            if old.ndim == 3: old = old.transpose(0,2,1)
            if width == 96: new = old.copy()
            else:
                new = np.array(value).copy()
                if name == 'encoder.weight': new[:96] = old
                elif name == 'encoder.bias': new[:96] = old
                elif name == 'output.weight':
                    new[:] = 0; new[:,:,:96] = old
                elif name == 'output.bias': new = old.copy()
                elif name.endswith('weight'):
                    # Preserve the original subnetwork exactly; added channels
                    # have independent random inputs/features, not copied replicas.
                    new[:96,:,:96] = old; new[:96,:,96:] = 0
                else: new[:96] = old
            weights.append((name,mx.array(new)))
        self.load_weights(weights,strict=False)
        self.freeze(keys=['mean','scale','left','fraction'],recurse=False)
    def hidden(self, mel):
        x = gelu(self.encoder(mel.transpose(0,2,1))); hints = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [5,7]:
                lo, hi = mx.take(x,self.left,axis=1), mx.take(x,self.left+1,axis=1)
                hints.append(lo+self.fraction*(hi-lo))
        return hints
    def __call__(self, mel): return self.output(self.hidden(mel)[-1])*self.scale+self.mean
    def torch_weights(self):
        return {name:(np.array(v).transpose(0,2,1).copy() if v.ndim==3 else np.array(v))
                for name,v in tree_flatten(self.trainable_parameters())}

class Learner(nn.Module):
    def __init__(self, values):
        super().__init__(); self.student=Predictor(values)
        self.heads=[nn.Linear(96,192),nn.Linear(96,192)]

def feature_loss(pred, target, valid):
    p = pred/mx.sqrt(mx.sum(pred*pred,axis=-1,keepdims=True)+1e-6)
    t = mx.stop_gradient(target/mx.sqrt(mx.sum(target*target,axis=-1,keepdims=True)+1e-6))
    return ((1-mx.sum(p*t,axis=-1))*valid).sum()/mx.maximum(valid.sum(),1)

def training_loss(net, mel, target, gaps, valid, hints, basis, rest, lips, coefficient):
    # One forward traversal; retain the existing unmodified animation objective.
    from mlx_backend import objective
    main=objective(net.student,mel,target,gaps,valid,basis,rest,lips)
    features=net.student.hidden(mel)
    hint=sum(feature_loss(head(x[:,15:45]),y,valid) for head,x,y in zip(net.heads,features,hints))/2
    return main+coefficient*hint

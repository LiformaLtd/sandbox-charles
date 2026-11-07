"""Native MLX training of the unchanged student; export remains standard ONNX."""
import os
os.environ['MLX_ENABLE_TF32']='0'
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def gelu(x):
    z=mx.clip(.7978845608028654*(x+.044715*x*x*x),-10.,10.)
    return .5*x*(1+mx.tanh(z))


class Block(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.conv=nn.Conv1d(96,96,3,padding=dilation,dilation=dilation);self.mix=nn.Conv1d(96,96,1)
    def __call__(self,x):return x+.5*self.mix(gelu(self.conv(x)))


class Student(nn.Module):
    def __init__(self,values):
        super().__init__();self.encoder=nn.Conv1d(64,96,5,padding=2);self.blocks=[Block(d) for d in [1,2,4,8,16,1,2,4]];self.output=nn.Conv1d(96,39,1)
        self.mean=mx.array(values['student.mean']);self.scale=mx.array(values['student.scale'])
        self.left=mx.array(values['student.left'].astype(np.int32));self.fraction=mx.array(values['student.fraction'].transpose(0,2,1))
        weights=[]
        for name,tensor in tree_flatten(self.parameters()):
            if name in ['mean','scale','left','fraction']:continue
            value=values['student.'+name]
            weights.append((name,mx.array(value.transpose(0,2,1) if value.ndim==3 else value)))
        self.load_weights(weights,strict=False);self.freeze(keys=['mean','scale','left','fraction'],recurse=False)
        if sum(v.size for _,v in tree_flatten(self.trainable_parameters()))!=331047:raise ValueError('Unexpected trainable parameter count')
    def __call__(self,mel):
        encoded=gelu(self.encoder(mel.transpose(0,2,1)))
        for block in self.blocks:encoded=block(encoded)
        lo=mx.take(encoded,self.left,axis=1);hi=mx.take(encoded,self.left+1,axis=1)
        return self.output(lo+self.fraction*(hi-lo))*self.scale+self.mean
    def torch_weights(self):
        result={}
        for name,value in tree_flatten(self.trainable_parameters()):
            v=np.array(value)
            result[name]=v.transpose(0,2,1).copy() if v.ndim==3 else v.copy()
        return result


def median9(x):
    indices=mx.argsort(mx.stop_gradient(x),axis=-1)[...,4:5]
    return mx.take_along_axis(x,indices,axis=-1)[...,0]


def smooth(error,beta):
    a=mx.abs(error);return mx.where(a<beta,.5*error*error/beta,a-.5*beta)


def objective(net,mel,target,gaps,valid,basis,rest,lips):
    pred=net(mel)[:,15:45];w=valid[...,None];count=mx.maximum(valid.sum(),1)
    shape=(smooth((pred-target)/net.scale,.5)*w*lips).sum()/(count*39)
    geom=mx.clip(pred,0,1)@basis+rest
    gap=(smooth(geom-gaps,.14)*w).sum()/(count*9)
    adjacent=valid[:,1:]&valid[:,:-1]
    velocity=(mx.abs((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1]))/net.scale*adjacent[...,None]).sum()/(count*39)
    actual,estimate=median9(gaps),median9(geom);contact=actual<=.1;logits=(.1-estimate)/.16
    bce=mx.maximum(logits,0)-logits*contact+mx.log1p(mx.exp(-mx.abs(logits)))
    closure=(bce*(1+contact)*valid).sum()/count
    near=((actual[:,1:]<.35)|(actual[:,:-1]<.35))&adjacent
    timing=(smooth((estimate[:,1:]-estimate[:,:-1])-(actual[:,1:]-actual[:,:-1]),.1)*near).sum()/mx.maximum(near.sum(),1)
    return shape+1.2*gap+.12*velocity+.22*closure+.45*timing

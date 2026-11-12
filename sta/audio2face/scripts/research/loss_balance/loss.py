"""Only fitted-coefficient SmoothL1 weight differs between the two arms."""
from common import *
import mlx.core as mx
from mlx_backend import smooth,median9

def components(net,mel,target,gaps,valid,basis,rest,lips):
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
 return shape,gap,velocity,closure,timing

def objective(net,mel,target,gaps,valid,basis,rest,lips,coefficient):
 shape,gap,velocity,closure,timing=components(net,mel,target,gaps,valid,basis,rest,lips)
 return coefficient*shape+1.2*gap+.12*velocity+.22*closure+.45*timing

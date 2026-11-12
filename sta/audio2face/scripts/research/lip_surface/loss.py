"""Exact dense surface MSE; training only, with no new model parameters."""
from common import *
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import mlx.core as mx
from mlx_backend import objective

def axis_squared(pred,stats,gram):
 q=stats[...,:117].reshape(*stats.shape[:-1],3,39)
 return mx.stack([mx.sum(pred*(pred@gram[a]),axis=-1)-2*mx.sum(pred*q[...,a,:],axis=-1)+stats[...,117+a] for a in range(3)],axis=-1)
def surface_loss(net,mel,stats,valid,gram):
 pred=mx.clip(net(mel)[:,15:45],0,1);squared=mx.maximum(mx.sum(axis_squared(pred,stats,gram),axis=-1),0)
 return (squared*valid).sum()/mx.maximum(valid.sum(),1)
def combined(net,mel,target,gaps,valid,stats,gram,basis,rest,lips,coefficient):
 return objective(net,mel,target,gaps,valid,basis,rest,lips)+coefficient*surface_loss(net,mel[:48],stats,valid[:48],gram)

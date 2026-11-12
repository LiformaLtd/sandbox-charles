"""Training-only hint projection. Export only student, never this wrapper."""
import os
os.environ['MLX_ENABLE_TF32']='0'
from common import *
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import mlx.core as mx,mlx.nn as nn
from mlx_backend import Student,gelu,objective
class TrainingModel(nn.Module):
 def __init__(self,values):
  super().__init__();self.student=Student(values);self.hint_head=nn.Linear(96,768)
 def hidden(self,mel):
  s=self.student;x=gelu(s.encoder(mel.transpose(0,2,1)))
  for block in s.blocks:x=block(x)
  lo=mx.take(x,s.left,axis=1);hi=mx.take(x,s.left+1,axis=1)
  return (lo+s.fraction*(hi-lo))[:,15:45]
 def hints(self,mel):return self.hint_head(self.hidden(mel))
def cosine_loss(pred,target,valid):
 p=pred/mx.sqrt(mx.sum(pred*pred,axis=-1,keepdims=True)+1e-6)
 t=target/mx.sqrt(mx.sum(target*target,axis=-1,keepdims=True)+1e-6)
 return ((1-mx.sum(p*t,axis=-1))*valid).sum()/mx.maximum(valid.sum(),1)
def loss(net,mel,weights,gaps,valid,hints,hint_valid,basis,rest,lips,coefficient):
 main=objective(net.student,mel,weights,gaps,valid,basis,rest,lips)
 auxiliary=cosine_loss(net.hints(mel[:48]),hints,hint_valid)
 return main+coefficient*auxiliary

"""Same spectral transform initially; learn only nonnegative mel-filter response."""
from common import *
import numpy as np
import mlx.core as mx,mlx.nn as nn
from mlx_backend import Student,objective

class Frontend(nn.Module):
 def __init__(self,bank,learn):
  super().__init__();self.base=mx.array(bank);self.weights=mx.array(bank)
  self.learn=learn
  self.freeze(keys=['base'],recurse=False)
 def bank(self):return self.weights if self.learn else mx.stop_gradient(self.weights)
 def project(self):self.weights=mx.maximum(self.weights,0)
 def __call__(self,power):
  # Projection occurs after updates, so zero-boundary coefficients can receive
  # gradients. Applying ReLU inside this forward pass would block those gradients.
  energy=power.transpose(0,2,1)@self.bank().T
  return mx.log1p(10*energy).transpose(0,2,1)

class Model(nn.Module):
 def __init__(self,values,learn):
  super().__init__();self.student=Student(values);self.frontend=Frontend(values['mel'][:,:,0],learn)
 @property
 def scale(self):return self.student.scale
 def __call__(self,power):return self.student(self.frontend(power))

def loss(net,power,target,gaps,valid,basis,rest,lips):
 return objective(net,power,target,gaps,valid,basis,rest,lips)

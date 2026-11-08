"""Jointly trainable v4 backbone and small causal frame-rate GRU correction."""
from common import *
import numpy as np
import mlx.core as mx,mlx.nn as nn
from mlx_backend import Student,gelu,smooth,median9
class Model(nn.Module):
 def __init__(self,values,recurrent=True):
  super().__init__();self.student=Student(values);self.cell=nn.GRU(96,32);self.head=nn.Linear(32,39);self.head.weight=mx.zeros_like(self.head.weight);self.head.bias=mx.zeros_like(self.head.bias);self.recurrent=recurrent
 def encoded(self,mel):
  s=self.student;x=gelu(s.encoder(mel.transpose(0,2,1)))
  for b in s.blocks:x=b(x)
  lo=mx.take(x,s.left,axis=1);hi=mx.take(x,s.left+1,axis=1)
  return (lo+s.fraction*(hi-lo))[:,15:45]
 def __call__(self,mel,valid,hidden):
  # mel [batch,4,64,101], valid [batch,120]. Never advance padding.
  n,k=mel.shape[:2];x=self.encoded(mel.reshape(n*k,64,101)).reshape(n,k*30,96)
  base=self.student.output(x)*self.student.scale+self.student.mean
  xp=mx.tanh(x/4)@self.cell.Wx.T+self.cell.b;outputs=[]
  for t in range(x.shape[1]):
   h=hidden if self.recurrent else mx.zeros_like(hidden)
   hp=h@self.cell.Wh.T;r=mx.sigmoid(xp[:,t,:32]+hp[:,:32]);z=mx.sigmoid(xp[:,t,32:64]+hp[:,32:64]);v=mx.tanh(xp[:,t,64:]+r*(hp[:,64:]+self.cell.bhn));next_h=(1-z)*v+z*h
   hidden=mx.where(valid[:,t,None],next_h,hidden);outputs.append(hidden)
  state=mx.stack(outputs,1)
  return base+self.head(state)*self.student.scale,hidden

def objective(net,mel,target,gaps,valid,hidden,basis,rest,lips):
 pred,state=net(mel,valid,hidden);scale=net.student.scale;w=valid[...,None];count=mx.maximum(valid.sum(),1)
 shape=(smooth((pred-target)/scale,.5)*w*lips).sum()/(count*39)
 geom=mx.clip(pred,0,1)@basis+rest;gap=(smooth(geom-gaps,.14)*w).sum()/(count*9)
 adjacent=valid[:,1:]&valid[:,:-1]
 velocity=(mx.abs((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1]))/scale*adjacent[...,None]).sum()/(count*39)
 actual,estimate=median9(gaps),median9(geom);contact=actual<=.1;logits=(.1-estimate)/.16
 bce=mx.maximum(logits,0)-logits*contact+mx.log1p(mx.exp(-mx.abs(logits)));closure=(bce*(1+contact)*valid).sum()/count
 near=((actual[:,1:]<.35)|(actual[:,:-1]<.35))&adjacent
 timing=(smooth((estimate[:,1:]-estimate[:,:-1])-(actual[:,1:]-actual[:,:-1]),.1)*near).sum()/mx.maximum(near.sum(),1)
 return shape+1.2*gap+.12*velocity+.22*closure+.45*timing,state

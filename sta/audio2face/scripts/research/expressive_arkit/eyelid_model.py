"""Separate learned lid tempo: six Hz squint/widen, twelve Hz blink anchors."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import OUT
from gaze import NEW
from full_neutral_model import NaturalFace, NaturalWaveform, conditions

LIDS=[0,5,6,7,12,13]

class LidDecoder(nn.Module):
    def __init__(self,mean,scale,means,scales,indices,stride):
        super().__init__();self.stride=stride
        self.register_buffer('mean',torch.as_tensor(mean).reshape(1,1,96))
        self.register_buffer('scale',torch.as_tensor(scale).reshape(1,1,96))
        self.register_buffer('means',torch.as_tensor(means[:,indices],dtype=torch.float32))
        self.register_buffer('scales',torch.as_tensor(scales[:,indices],dtype=torch.float32))
        self.input=nn.Conv1d(107,32,1);self.conv1=nn.Conv1d(32,32,3,padding=1);self.conv2=nn.Conv1d(32,32,3,padding=1)
        self.output=nn.Conv1d(32,len(indices),1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,hidden,emotion):
        control=conditions(emotion)
        x=torch.cat([(hidden-self.mean)/self.scale,control],dim=-1).transpose(1,2)
        x=F.avg_pool1d(x,self.stride,self.stride)
        x=F.gelu(self.input(x),approximate='tanh')
        x=x+.5*F.gelu(self.conv1(x),approximate='tanh');x=x+.5*F.gelu(self.conv2(x),approximate='tanh')
        residual=F.interpolate(self.output(x),size=60,mode='linear',align_corners=False).transpose(1,2)
        return (control@self.means+(control@self.scales)*residual).clamp(0,1)

class StableLids(nn.Module):
    def __init__(self,parent_dir,means,scales):
        super().__init__()
        adapter=torch.load(OUT/'adapter.pt',map_location='cpu',weights_only=True)
        gaze=torch.load(NEW/'gaze.pt',map_location='cpu',weights_only=True)
        with np.load(parent_dir/'brow-statistics.npz') as z:
            self.parent=NaturalFace(adapter,gaze,z['means'],z['scales'])
        self.parent.load_state_dict(torch.load(parent_dir/'face.pt',map_location='cpu',weights_only=True));self.parent.requires_grad_(False)
        self.slow=LidDecoder(adapter['feature_mean'],adapter['feature_scale'],means,scales,[1,2,4,5],10)
        self.blink=LidDecoder(adapter['feature_mean'],adapter['feature_scale'],means,scales,[0,3],5)
        mask=torch.ones(52);mask[LIDS]=0;self.register_buffer('mask',mask)
        self.register_buffer('slow_map',F.one_hot(torch.tensor([5,6,12,13]),52).float())
        self.register_buffer('blink_map',F.one_hot(torch.tensor([0,7]),52).float())
    def lids(self,hidden,emotion):
        return self.slow(hidden,emotion)@self.slow_map+self.blink(hidden,emotion)@self.blink_map
    def forward(self,hidden,speech,emotion,style):
        return self.parent(hidden,speech,emotion,style)*self.mask+self.lids(hidden,emotion)

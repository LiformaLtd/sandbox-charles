"""Learned mouth residual with teacher-rest supervision; frozen upper-face tracks."""
import torch
from torch import nn
from torch.nn import functional as F
from common import *
from mouth_corner_model import MouthCornerFace

CURRENT=APP/'.local/mouth-corner-isolated-v1'
DEST=APP/'.local/speech-rest-v1'

class SpeechRestFace(nn.Module):
    def __init__(self):
        super().__init__();self.base=MouthCornerFace();self.base.load_state_dict(torch.load(CURRENT/'face.pt',map_location='cpu',weights_only=True));self.base.requires_grad_(False)
        self.register_buffer('feature_mean',self.base.base.fast.feature_mean.clone())
        self.register_buffer('feature_scale',self.base.base.fast.feature_scale.clone())
        self.input=nn.Conv1d(133,32,5);self.conv1=nn.Conv1d(32,32,3);self.conv2=nn.Conv1d(32,32,3);self.output=nn.Conv1d(32,27,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def correct(self,hidden,base,emotion):
        x=torch.cat([(hidden-self.feature_mean)/self.feature_scale,base[:,:,14:41],emotion],dim=-1).transpose(1,2)
        x=F.gelu(self.input(F.pad(x,(2,2),mode='replicate')),approximate='tanh')
        x=x+.5*F.gelu(self.conv1(F.pad(x,(1,1),mode='replicate')),approximate='tanh')
        x=x+.5*F.gelu(self.conv2(F.pad(x,(1,1),mode='replicate')),approximate='tanh')
        mouth=(base[:,:,14:41]+self.output(x).transpose(1,2)).clamp(0,1)
        return torch.cat([base[:,:,:14],mouth,base[:,:,41:]],dim=-1)
    def forward(self,hidden,speech,emotion,style):
        return self.correct(hidden,self.base(hidden,speech,emotion,style),emotion)

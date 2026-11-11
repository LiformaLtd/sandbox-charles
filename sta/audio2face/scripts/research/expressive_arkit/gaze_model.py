"""Condition gaze on teacher identity style without averaging opposite directions."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import ExpressiveWaveform


class GazeHead(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.register_buffer('feature_mean',torch.as_tensor(mean).reshape(1,1,96))
        self.register_buffer('feature_scale',torch.as_tensor(scale).reshape(1,1,96))
        self.input=nn.Conv1d(109,32,5,padding=2)
        self.conv1=nn.Conv1d(32,32,3,padding=1)
        self.conv2=nn.Conv1d(32,32,3,padding=4,dilation=4)
        self.output=nn.Conv1d(32,4,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        mapping=F.one_hot(torch.tensor([8,9,1,3,11,10,4,2]),52).float()
        self.register_buffer('channel_map',mapping)

    def forward(self,hidden,emotion,style):
        x=torch.cat([(hidden-self.feature_mean)/self.feature_scale,emotion,style],dim=-1).transpose(1,2)
        x=F.gelu(self.input(x),approximate='tanh')
        x=x+.5*F.gelu(self.conv1(x),approximate='tanh')
        x=x+.5*F.gelu(self.conv2(x),approximate='tanh')
        return self.output(x).transpose(1,2)

    def coefficients(self,normalized_angles):
        signed=torch.cat([normalized_angles,-normalized_angles],dim=-1)
        return signed.clamp(0,1)@self.channel_map


class FullFaceWaveform(nn.Module):
    def __init__(self,adapter,gaze):
        super().__init__()
        self.face=ExpressiveWaveform(adapter)
        self.face.requires_grad_(False)
        self.gaze=gaze
        mask=torch.ones(52);mask[[1,2,3,4,8,9,10,11]]=0
        self.register_buffer('face_mask',mask)

    def forward(self,pcm,emotion,gaze_style):
        return self.forward_tracks(pcm,emotion,gaze_style)[0]

    def forward_tracks(self,pcm,emotion,gaze_style):
        speech=self.face.speech
        spectrum=F.conv1d(pcm.unsqueeze(1),speech.filters,stride=160,padding=200)
        real,imag=spectrum.chunk(2,dim=1)
        mel=(1+F.conv1d(real.square()+imag.square(),speech.mel)*10).log()
        s=speech.student
        hidden=s.blocks(F.gelu(s.encoder(mel),approximate='tanh'))
        aligned=hidden.index_select(2,s.left)+s.fraction*(hidden.index_select(2,s.left+1)-hidden.index_select(2,s.left))
        raw=s.output(aligned).transpose(1,2)*s.scale+s.mean
        coefficients=raw.clamp(0,1)@s.channel_map
        hidden=aligned.transpose(1,2)
        face=self.face.adapter(hidden,coefficients,emotion)
        angles=self.gaze(hidden,emotion,gaze_style)
        return face*self.face_mask+self.gaze.coefficients(angles),coefficients


class FullFaceWithArticulation(nn.Module):
    """Expose the unchanged speech track for future per-avatar contact handling."""
    def __init__(self,model):
        super().__init__()
        self.model=model

    def forward(self,pcm,emotion,gaze_style):
        return self.model.forward_tracks(pcm,emotion,gaze_style)

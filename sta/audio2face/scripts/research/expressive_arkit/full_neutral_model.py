"""Teacher-driven neutral face, fast articulation, and a slow learned brow path."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import ExpressionAdapter
from gaze_model import GazeHead
from prepare import baseline


def conditions(emotion):
    total=emotion.sum(-1,keepdim=True)
    normalized=emotion/total.clamp_min(1)
    return torch.cat([(1-total).clamp(0,1),normalized],dim=-1)


class NaturalBrows(nn.Module):
    def __init__(self,feature_mean,feature_scale,means,scales):
        super().__init__()
        self.register_buffer('feature_mean',torch.as_tensor(feature_mean).reshape(1,1,96))
        self.register_buffer('feature_scale',torch.as_tensor(feature_scale).reshape(1,1,96))
        self.register_buffer('means',torch.as_tensor(means,dtype=torch.float32))
        self.register_buffer('scales',torch.as_tensor(scales,dtype=torch.float32))
        self.input=nn.Conv1d(107,32,1)
        self.conv1=nn.Conv1d(32,32,3,padding=1)
        self.conv2=nn.Conv1d(32,32,3,padding=1)
        self.output=nn.Conv1d(32,5,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,emotion):
        control=conditions(emotion)
        x=torch.cat([(hidden-self.feature_mean)/self.feature_scale,control],dim=-1).transpose(1,2)
        x=F.avg_pool1d(x,10,10)
        x=F.gelu(self.input(x),approximate='tanh')
        x=x+.5*F.gelu(self.conv1(x),approximate='tanh')
        x=x+.5*F.gelu(self.conv2(x),approximate='tanh')
        residual=F.interpolate(self.output(x),size=60,mode='linear',align_corners=False).transpose(1,2)
        return ((control@self.means)+(control@self.scales)*residual).clamp(0,1)


class NaturalFace(nn.Module):
    def __init__(self,adapter_state,gaze_state,brow_means,brow_scales):
        super().__init__()
        protected=np.flatnonzero(adapter_state['speech_owned'].numpy())
        self.fast=ExpressionAdapter(protected,np.zeros(96,np.float32),np.ones(96,np.float32))
        self.fast.load_state_dict(adapter_state)
        # Begin with the proven articulation prior, but every facial correction
        # is trainable at neutral. This is initialization, not an inference gate.
        nn.init.zeros_(self.fast.output.weight);nn.init.zeros_(self.fast.output.bias)
        self.brows=NaturalBrows(adapter_state['feature_mean'],adapter_state['feature_scale'],brow_means,brow_scales)
        self.gaze=GazeHead(adapter_state['feature_mean'],adapter_state['feature_scale'])
        self.gaze.load_state_dict(gaze_state);self.gaze.requires_grad_(False)
        mask=torch.ones(52);mask[[1,2,3,4,8,9,10,11,41,42,43,44,45]]=0
        self.register_buffer('fast_mask',mask)
        self.register_buffer('brow_map',F.one_hot(torch.arange(41,46),52).float())

    def forward(self,hidden,speech,emotion,style):
        context=(hidden-self.fast.feature_mean)/self.fast.feature_scale
        x=torch.cat([context,speech,emotion],dim=-1).transpose(1,2)
        latent=self.fast.blocks(F.gelu(self.fast.input(x),approximate='tanh'))
        correction=self.fast.output(latent).transpose(1,2)
        face=(speech+correction).clamp(0,1)
        brows=self.brows(hidden,emotion)
        eyes=self.gaze.coefficients(self.gaze(hidden,emotion,style))
        return face*self.fast_mask+brows@self.brow_map+eyes


class NaturalWaveform(nn.Module):
    def __init__(self,face):
        super().__init__()
        source,student,_=baseline()
        self.speech=source.WaveformModel(student);self.speech.requires_grad_(False)
        self.face=face

    def forward(self,pcm,emotion,gaze_style):
        spectrum=F.conv1d(pcm.unsqueeze(1),self.speech.filters,stride=160,padding=200)
        real,imag=spectrum.chunk(2,dim=1)
        mel=(1+F.conv1d(real.square()+imag.square(),self.speech.mel)*10).log()
        s=self.speech.student
        h=s.blocks(F.gelu(s.encoder(mel),approximate='tanh'))
        aligned=h.index_select(2,s.left)+s.fraction*(h.index_select(2,s.left+1)-h.index_select(2,s.left))
        raw=s.output(aligned).transpose(1,2)*s.scale+s.mean
        articulation=raw.clamp(0,1)@s.channel_map
        return self.face(aligned.transpose(1,2),articulation,emotion,gaze_style),articulation

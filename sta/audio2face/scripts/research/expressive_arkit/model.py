"""Frozen articulation plus a small audio/emotion-conditioned full-face adapter."""
import torch
from torch import nn
from torch.nn import functional as F
from prepare import baseline


class Residual(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.conv=nn.Conv1d(64,64,3,padding=dilation,dilation=dilation)
        self.mix=nn.Conv1d(64,64,1)

    def forward(self,x):
        return x+.5*self.mix(F.gelu(self.conv(x),approximate='tanh'))


class ExpressionAdapter(nn.Module):
    def __init__(self, active, mean, scale):
        super().__init__()
        self.register_buffer('feature_mean',torch.as_tensor(mean).view(1,1,96))
        self.register_buffer('feature_scale',torch.as_tensor(scale).view(1,1,96))
        owned=torch.zeros(52)
        owned[torch.as_tensor(active)]=1
        self.register_buffer('speech_owned',owned)
        self.input=nn.Conv1d(158,64,5,padding=2)
        self.blocks=nn.Sequential(*(Residual(d) for d in [1,2,8]))
        self.output=nn.Conv1d(64,52,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,hidden,speech,emotion):
        context=(hidden-self.feature_mean)/self.feature_scale
        x=torch.cat([context,speech,emotion],dim=-1).transpose(1,2)
        delta=self.output(self.blocks(F.gelu(self.input(x),approximate='tanh'))).transpose(1,2)
        # Hard neutral protection for every previously learned coefficient.
        # Newly supervised gaze/lateral/tongue channels may move at neutral.
        strength=emotion.amax(dim=-1,keepdim=True).clamp(0,1)
        gate=1-self.speech_owned+self.speech_owned*strength
        return (speech+delta*gate).clamp(0,1)


class ExpressiveWaveform(nn.Module):
    def __init__(self,adapter):
        super().__init__()
        source,student,_=baseline()
        self.speech=source.WaveformModel(student)
        self.speech.requires_grad_(False)
        self.adapter=adapter

    def forward(self,pcm,emotion):
        spectrum=F.conv1d(pcm.unsqueeze(1),self.speech.filters,stride=160,padding=200)
        real,imag=spectrum.chunk(2,dim=1)
        mel=(1+F.conv1d(real.square()+imag.square(),self.speech.mel)*10).log()
        s=self.speech.student
        hidden=s.blocks(F.gelu(s.encoder(mel),approximate='tanh'))
        aligned=hidden.index_select(2,s.left)+s.fraction*(hidden.index_select(2,s.left+1)-hidden.index_select(2,s.left))
        raw=s.output(aligned).transpose(1,2)*s.scale+s.mean
        speech=raw.clamp(0,1)@s.channel_map
        return self.adapter(aligned.transpose(1,2),speech,emotion)

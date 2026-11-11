"""Energy-aware learned rest correction; no runtime VAD threshold or end timer."""
import torch
from torch import nn
from torch.nn import functional as F
from common import *
from mouth_corner_model import MouthCornerFace
from refine_mouth_tails import NaturalWaveform

CURRENT=APP/'.local/mouth-corner-isolated-v1'
DEST=APP/'.local/speech-rest-v3'

class EnergyRestFace(nn.Module):
    def __init__(self):
        super().__init__();self.base=MouthCornerFace();self.base.load_state_dict(torch.load(CURRENT/'face.pt',map_location='cpu',weights_only=True));self.base.requires_grad_(False)
        self.register_buffer('feature_mean',self.base.base.fast.feature_mean.clone());self.register_buffer('feature_scale',self.base.base.fast.feature_scale.clone())
        self.input=nn.Conv1d(136,48,5);self.conv1=nn.Conv1d(48,48,3);self.conv2=nn.Conv1d(48,48,3);self.output=nn.Conv1d(48,27,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def correct(self,hidden,base,emotion,energy):
        x=torch.cat([(hidden-self.feature_mean)/self.feature_scale,base[:,:,14:41],emotion,energy],dim=-1).transpose(1,2)
        x=F.gelu(self.input(F.pad(x,(2,2),mode='replicate')),approximate='tanh')
        x=x+.5*F.gelu(self.conv1(F.pad(x,(1,1),mode='replicate')),approximate='tanh');x=x+.5*F.gelu(self.conv2(F.pad(x,(1,1),mode='replicate')),approximate='tanh')
        mouth=(base[:,:,14:41]+self.output(x).transpose(1,2)).clamp(0,1)
        return torch.cat([base[:,:,:14],mouth,base[:,:,41:]],dim=-1)

class EnergyRestWaveform(NaturalWaveform):
    def __init__(self,face):
        super().__init__(face)
        for size in [320,1600,3200]:self.register_buffer('energy_'+str(size),torch.ones(1,1,size)/size)
    def energy(self,pcm):
        s=self.speech.student;values=[]
        for size in [320,1600,3200]:
            power=F.conv1d(pcm.unsqueeze(1).square(),getattr(self,'energy_'+str(size)),stride=160,padding=size//2)
            aligned=power.index_select(2,s.left)+s.fraction*(power.index_select(2,s.left+1)-power.index_select(2,s.left))
            values.append((1+aligned*1e6).log())
        return torch.cat(values,dim=1).transpose(1,2)
    def forward(self,pcm,emotion,gaze_style):
        spectrum=F.conv1d(pcm.unsqueeze(1),self.speech.filters,stride=160,padding=200);real,imag=spectrum.chunk(2,dim=1)
        mel=(1+F.conv1d(real.square()+imag.square(),self.speech.mel)*10).log();s=self.speech.student
        h=s.blocks(F.gelu(s.encoder(mel),approximate='tanh'));aligned=h.index_select(2,s.left)+s.fraction*(h.index_select(2,s.left+1)-h.index_select(2,s.left))
        hidden=aligned.transpose(1,2);speech=(s.output(aligned).transpose(1,2)*s.scale+s.mean).clamp(0,1)@s.channel_map
        base=self.face.base(hidden,speech,emotion,gaze_style)
        return self.face.correct(hidden,base,emotion,self.energy(pcm)),speech

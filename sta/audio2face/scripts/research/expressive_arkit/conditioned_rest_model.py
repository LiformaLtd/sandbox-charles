"""Continuous audio-conditioned rest poses inside the research ONNX graph.

A soft energy response suppresses the decoder's response to quiet-window context;
there is no clip-end timer or binary VAD decision. Speech core and 25 other face
controls are retained. Rest poses are fitted against teacher geometry.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common import *
from speech_rest_energy_model import EnergyRestFace,EnergyRestWaveform,CURRENT
DEST=APP/'.local/conditioned-rest-v3'

class ConditionedRestWaveform(EnergyRestWaveform):
    def __init__(self):
        super().__init__(EnergyRestFace())
        self.requires_grad_(False)
        self.register_buffer('rest_energy',torch.ones(1,1,6400)/6400)
        self.rest=nn.Parameter(torch.zeros(11,27))
    def activity(self,pcm):
        s=self.speech.student
        power=F.conv1d(F.pad(pcm.unsqueeze(1).square(),(1280,5120)),self.rest_energy,stride=160)
        aligned=power.index_select(2,s.left)+s.fraction*(power.index_select(2,s.left+1)-power.index_select(2,s.left))
        return (1-torch.exp(-(1+aligned*1e6).log()/.1)).transpose(1,2)
    def forward(self,pcm,emotion,gaze_style):
        base,articulation=super().forward(pcm,emotion,gaze_style)
        activity=self.activity(pcm)
        rest=(self.rest[0]+emotion@(self.rest[1:]-self.rest[0])).clamp(0,1)
        mouth=base[:,:,14:41]*activity+rest*(1-activity)
        return torch.cat([base[:,:,:14],mouth,base[:,:,41:]],dim=-1),articulation

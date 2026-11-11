"""Continuous quiet-pose conditioning inside the research ONNX graph.

Quiet-window context caused periodic jaw/chin movement. A small fitted affine
rest decoder uses each window's quiet-weighted mouth pose. Audio energy blends
speech and rest continuously; there is no binary VAD or clip-end timer. The
speech core, 25 non-mouth controls and animation sample grid stay unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common import APP
from mouth_corner_model import MouthCornerFace
from refine_mouth_tails import NaturalWaveform
CURRENT=APP/'.local/mouth-corner-isolated-v1'
DEST=APP/'.local/adaptive-rest-v2'

class AdaptiveRestWaveform(NaturalWaveform):
    def __init__(self):
        face=MouthCornerFace()
        face.load_state_dict(torch.load(CURRENT/'face.pt',map_location='cpu',weights_only=True))
        super().__init__(face)
        self.requires_grad_(False)
        self.register_buffer('rest_energy',torch.ones(1,1,6400)/6400)
        self.rest=nn.Parameter(torch.zeros(11,27))
        self.rest_transform=nn.Linear(27,27,bias=False)
        nn.init.zeros_(self.rest_transform.weight)

    def load_rest(self,path):
        """Load just the trained adapter; frozen speech/face sources stay pinned."""
        state=torch.load(path,map_location='cpu',weights_only=True)
        self.rest.data.copy_(state['rest'])
        self.rest_transform.weight.data.copy_(state['rest_transform.weight'])

    def activity(self,pcm):
        s=self.speech.student
        # 80 ms past / 320 ms future supports anticipatory articulation without
        # retaining a 200 ms response after speech has finished.
        power=F.conv1d(F.pad(pcm.unsqueeze(1).square(),(1280,5120)),self.rest_energy,stride=160)
        aligned=power.index_select(2,s.left)+s.fraction*(power.index_select(2,s.left+1)-power.index_select(2,s.left))
        return (1-torch.exp(-(1+aligned*1e6).log()/.1)).transpose(1,2)

    def rest_pose(self,base,activity,emotion):
        quiet=1-activity
        mean=(base[:,:,14:41]*quiet).sum(1,keepdim=True)/quiet.sum(1,keepdim=True).clamp_min(1e-6)
        delta=self.rest[0]+emotion@(self.rest[1:]-self.rest[0])
        return (mean+self.rest_transform(mean)+delta).clamp(0,1)

    def forward(self,pcm,emotion,gaze_style):
        base,articulation=super().forward(pcm,emotion,gaze_style)
        activity=self.activity(pcm)
        mouth=base[:,:,14:41]*activity+self.rest_pose(base,activity,emotion)*(1-activity)
        return torch.cat([base[:,:,:14],mouth,base[:,:,41:]],dim=-1),articulation

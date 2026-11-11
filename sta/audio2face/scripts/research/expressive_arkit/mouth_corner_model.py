"""Correct mouth poses without borrowing cheek/nose/eyelid motion."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import *
from guard_mouth_tails import GapGuardFace
from fit import mm

CURRENT=APP/'.local/mouth-tail-gap-guard-v1'
TRAINED=APP/'.local/mouth-tail-gap-guard-v3'
DEST=APP/'.local/mouth-corner-isolated-v1'

class MouthCornerFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=GapGuardFace();self.base.load_state_dict(torch.load(CURRENT/'face.pt',map_location='cpu',weights_only=True));self.base.requires_grad_(False)
        trained=GapGuardFace();trained.load_state_dict(torch.load(TRAINED/'face.pt',map_location='cpu',weights_only=True))
        for name,value in self.base.fast.state_dict().items():
            if not name.startswith('output.') and not torch.equal(value,trained.fast.state_dict()[name]):raise ValueError('Output-only refinement changed shared feature/body weights')
        self.output=copy.deepcopy(trained.fast.output);self.output.requires_grad_(False)
        indices=torch.arange(14,41);self.register_buffer('mouth_indices',indices)
        self.register_buffer('mouth_map',F.one_hot(indices,52).float())
        mask=torch.ones(52);mask[14:41]=0;self.register_buffer('mouth_keep',mask)
        with np.load(OUT/'calibration.npz') as z:g=z['gap_basis'][14:41].astype(np.float64)
        u,s,_=np.linalg.svd(g,full_matrices=False);rank=int((s>s[0]*1e-8).sum())
        self.register_buffer('mouth_projection',torch.from_numpy(mm(u[:,:rank],u[:,:rank].T).astype(np.float32)))
    def project(self,desired,reference,projection,iterations):
        x=desired;p=torch.zeros_like(x);q=torch.zeros_like(x)
        for _ in range(iterations):
            value=x+p;y=value-(value-reference)@projection;p=value-y
            value=y+q;x=value.clamp(0,1);q=value-x
        return x
    def forward(self,hidden,speech,emotion,style):
        original=self.base.parent(hidden,speech,emotion,style)
        h=(hidden-self.base.fast.feature_mean)/self.base.fast.feature_scale
        latent=self.base.fast.blocks(F.gelu(self.base.fast.input(torch.cat([h,speech,emotion],dim=-1).transpose(1,2)),approximate='tanh'))
        desired=(speech+self.base.fast.output(latent).transpose(1,2)).clamp(0,1).index_select(-1,self.base.active)
        current=self.project(desired,original.index_select(-1,self.base.active),self.base.normal_projection,128)
        full=original*self.base.keep_mask+current@self.base.channel_map
        refined=(speech+self.output(latent).transpose(1,2)).clamp(0,1).index_select(-1,self.mouth_indices)
        guarded=self.project(refined,full.index_select(-1,self.mouth_indices),self.mouth_projection,256)
        return full*self.mouth_keep+guarded@self.mouth_map

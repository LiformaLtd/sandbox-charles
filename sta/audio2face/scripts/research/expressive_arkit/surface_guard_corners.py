"""Constrain corner changes against lip gaps and dominant upper-face geometry."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import *
from guard_mouth_tails import GapGuardFace,ACTIVE
from fit import mm

CURRENT=APP/'.local/mouth-tail-gap-guard-v1'
TRAINED=APP/'.local/mouth-tail-gap-guard-v3'

class SurfaceGuardFace(nn.Module):
    def __init__(self,tolerance_mm=.005,iterations=256):
        super().__init__();self.iterations=iterations
        self.base=GapGuardFace();self.base.load_state_dict(torch.load(CURRENT/'face.pt',map_location='cpu',weights_only=True));self.base.requires_grad_(False)
        trained=GapGuardFace();trained.load_state_dict(torch.load(TRAINED/'face.pt',map_location='cpu',weights_only=True))
        self.output=copy.deepcopy(trained.fast.output);self.output.requires_grad_(False)
        with np.load(OUT/'calibration.npz') as z:shape=z['shape'];gap=z['gap_basis'][ACTIVE].astype(np.float64)
        brow=np.argsort(np.max(np.linalg.norm(shape[41:46],axis=2),axis=0))[-512:]
        lids=np.unique(np.concatenate([np.argsort(np.linalg.norm(shape[c],axis=1))[-256:] for c in [0,5,6,7,12,13]]))
        region=np.unique(np.concatenate([brow,lids]));basis=shape[ACTIVE][:,region].reshape(len(ACTIVE),-1).astype(np.float64)*10/np.sqrt(len(region)*3)
        upper,singular,_=np.linalg.svd(basis,full_matrices=False);keep=singular>tolerance_mm
        constraints=np.concatenate([gap/np.linalg.norm(gap,axis=0,keepdims=True),upper[:,keep]],axis=1)
        u,s,_=np.linalg.svd(constraints,full_matrices=False);rank=int((s>s[0]*1e-7).sum())
        self.rank=rank;self.upper_directions=int(keep.sum());self.tolerance_mm=tolerance_mm
        self.register_buffer('projection',torch.from_numpy(mm(u[:,:rank],u[:,:rank].T).astype(np.float32)))
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
        reference=original.index_select(-1,self.base.active)
        current=self.project(desired,reference,self.base.normal_projection,128)
        refined=(speech+self.output(latent).transpose(1,2)).clamp(0,1).index_select(-1,self.base.active)
        guarded=self.project(refined,current,self.projection,self.iterations)
        return original*self.base.keep_mask+guarded@self.base.channel_map

class MouthOnlyGuardFace(SurfaceGuardFace):
    def __init__(self,iterations=256):
        super().__init__(.02,iterations)
        indices=torch.arange(14,41);self.register_buffer('mouth_indices',indices)
        self.register_buffer('mouth_map',F.one_hot(indices,52).float())
        mask=torch.ones(52);mask[14:41]=0;self.register_buffer('mouth_keep',mask)
        with np.load(OUT/'calibration.npz') as z:g=z['gap_basis'][14:41].astype(np.float64)
        u,s,_=np.linalg.svd(g,full_matrices=False);rank=int((s>s[0]*1e-8).sum())
        self.register_buffer('mouth_projection',torch.from_numpy(mm(u[:,:rank],u[:,:rank].T).astype(np.float32)))
    def forward(self,hidden,speech,emotion,style):
        original=self.base.parent(hidden,speech,emotion,style)
        h=(hidden-self.base.fast.feature_mean)/self.base.fast.feature_scale
        latent=self.base.fast.blocks(F.gelu(self.base.fast.input(torch.cat([h,speech,emotion],dim=-1).transpose(1,2)),approximate='tanh'))
        desired=(speech+self.base.fast.output(latent).transpose(1,2)).clamp(0,1).index_select(-1,self.base.active)
        reference=original.index_select(-1,self.base.active)
        current=self.project(desired,reference,self.base.normal_projection,128)
        full=original*self.base.keep_mask+current@self.base.channel_map
        refined=(speech+self.output(latent).transpose(1,2)).clamp(0,1).index_select(-1,self.mouth_indices)
        guarded=self.project(refined,full.index_select(-1,self.mouth_indices),self.mouth_projection,self.iterations)
        return full*self.mouth_keep+guarded@self.mouth_map

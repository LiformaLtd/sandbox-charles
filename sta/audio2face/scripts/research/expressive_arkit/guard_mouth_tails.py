"""Bounded affine projection: refine corners while retaining parent lip gaps."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from common import *
from refine_mouth_tails import load_net,DEST as REFINED,PARENT,NaturalWaveform
from eyelid_model import LIDS
from fit import mm

DEST=APP/'.local/mouth-tail-gap-guard-v1'
ITERATIONS=128
ACTIVE=[c for c in range(52) if c not in [*LIDS,1,2,3,4,8,9,10,11,41,42,43,44,45]]

class GapGuardFace(nn.Module):
    def __init__(self):
        super().__init__();self.parent=load_net();self.parent.requires_grad_(False)
        trained=load_net();trained.load_state_dict(torch.load(REFINED/'face.pt',map_location='cpu',weights_only=True))
        self.fast=copy.deepcopy(trained.parent.fast);self.fast.requires_grad_(False)
        with np.load(OUT/'calibration.npz') as z:g=z['gap_basis'][ACTIVE].astype(np.float64)
        u,s,_=np.linalg.svd(g,full_matrices=False);rank=int((s>s[0]*1e-8).sum());q=mm(u[:,:rank],u[:,:rank].T).astype(np.float32)
        self.register_buffer('normal_projection',torch.from_numpy(q))
        self.register_buffer('active',torch.tensor(ACTIVE))
        mask=torch.ones(52);mask[ACTIVE]=0;self.register_buffer('keep_mask',mask)
        self.register_buffer('channel_map',F.one_hot(torch.tensor(ACTIVE),52).float())
    def forward(self,hidden,speech,emotion,style):
        parent=self.parent(hidden,speech,emotion,style)
        h=(hidden-self.fast.feature_mean)/self.fast.feature_scale
        latent=self.fast.blocks(F.gelu(self.fast.input(torch.cat([h,speech,emotion],dim=-1).transpose(1,2)),approximate='tanh'))
        desired=(speech+self.fast.output(latent).transpose(1,2)).clamp(0,1).index_select(-1,self.active)
        reference=parent.index_select(-1,self.active)
        # Dykstra projections onto the parent-gap affine subspace and [0,1]
        # coefficient box. Neither corner pose nor silence pose is prescribed.
        x=desired;p=torch.zeros_like(x);r=torch.zeros_like(x)
        for _ in range(ITERATIONS):
            value=x+p;y=value-(value-reference)@self.normal_projection
            p=value-y;value=y+r;x=value.clamp(0,1);r=value-x
        return parent*self.keep_mask+x@self.channel_map

def main():
    DEST.mkdir(exist_ok=True);torch.set_num_threads(4)
    if (DEST/'model.onnx').exists():raise ValueError('Guarded artifact exists')
    net=GapGuardFace().eval();torch.save(net.state_dict(),DEST/'face.pt')
    freeze(DEST/'protocol.json',dict(source_sha256=sha(__file__),parent_sha256=sha(PARENT/'model.onnx'),refined_sha256=sha(REFINED/'model.onnx'),
        iterations=ITERATIONS,method='Dykstra box/affine projection of refined active coefficients onto nine parent lip-gap constraints',
        scope='James canonical blendshape geometry; not a Player or 2.5D mapping change',
        acceptance='Same predeclared corner, join, lip and frozen-channel checks as the unguarded refinement'))
    model=NaturalWaveform(net).eval();style=torch.zeros(1,60,3);style[:,:,1]=1
    torch.onnx.export(model,(torch.zeros(1,16000),torch.zeros(1,60,10),style),str(DEST/'model.onnx'),
        input_names=['pcm','emotion','gaze_style'],output_names=['blendshapes','articulation'],opset_version=17,dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(DEST/'model.onnx'))
    print('EXPORTED',sha(DEST/'model.onnx'),(DEST/'model.onnx').stat().st_size,flush=True)
    import verify_mouth_tails as verify
    verify.DEST=DEST;verify.load_net=GapGuardFace;verify.main()

if __name__=='__main__':main()

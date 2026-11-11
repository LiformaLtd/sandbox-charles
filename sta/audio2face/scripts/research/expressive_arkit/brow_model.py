"""A learned low-rate decoder for brows; all non-brow outputs stay frozen."""
import torch
from torch import nn
from torch.nn import functional as F

BROW_IDS=[41,42,43,44,45]


class BrowDecoder(nn.Module):
    def __init__(self,weight,bias):
        super().__init__()
        self.projection=nn.Conv1d(64,5,1)
        self.emotion_bias=nn.Linear(10,5,bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.as_tensor(weight))
            self.projection.bias.copy_(torch.as_tensor(bias))
            self.emotion_bias.weight.zero_()

    def slow_latents(self,latent):
        # Six curve anchors per second. Group centres and half-second inference
        # hops have a shared temporal grid; decoding keeps the original 60 fps
        # sample positions and requires no additional audio lookahead.
        return F.avg_pool1d(latent,kernel_size=10,stride=10)

    def decode(self,slow,emotion):
        brows=F.interpolate(self.projection(slow),size=60,mode='linear',align_corners=False).transpose(1,2)
        return (brows+self.emotion_bias(emotion)).clamp(0,1)

    def forward(self,latent,emotion):
        return self.decode(self.slow_latents(latent),emotion)

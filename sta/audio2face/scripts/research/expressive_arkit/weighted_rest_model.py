"""Suppress window-context mouth movement during quiet audio inside ONNX.

Continuously weighted quiet pose retains each window's expression. Speech core,
25 other coefficients and the original animation sample grid are unchanged.
"""
import torch
from common import *
from conditioned_rest_model import ConditionedRestWaveform
DEST=APP/'.local/weighted-rest-v1'

class WeightedRestWaveform(ConditionedRestWaveform):
    def forward(self,pcm,emotion,gaze_style):
        base,articulation=super(ConditionedRestWaveform,self).forward(pcm,emotion,gaze_style)
        activity=self.activity(pcm);quiet=1-activity
        # Only quiet audio contributes to the window's settling pose.
        rest=(base[:,:,14:41]*quiet).sum(1,keepdim=True)/quiet.sum(1,keepdim=True).clamp_min(1e-6)
        mouth=base[:,:,14:41]*activity+rest*quiet
        return torch.cat([base[:,:,:14],mouth,base[:,:,41:]],dim=-1),articulation

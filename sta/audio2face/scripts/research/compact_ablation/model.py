"""Train a small streaming audio-to-James-blendshape student and export ONNX."""
from pathlib import Path
import argparse
import copy
import json
import math
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(6)
torch.manual_seed(37)
RATE, NFFT, WIN, HOP = 16000, 512, 400, 160
CHANNELS = 64
offsets = np.arange(15,45)*RATE//60

def mel_bank():
    hz_to_mel = lambda x: 2595 * np.log10(1 + x/700)
    edges = 700 * (10 ** (np.linspace(hz_to_mel(60),hz_to_mel(7900),CHANNELS+2)/2595) - 1)
    bins = np.fft.rfftfreq(NFFT, 1/RATE)
    up = (bins[None,:]-edges[:-2,None]) / (edges[1:-1]-edges[:-2])[:,None]
    down = (edges[2:,None]-bins[None,:]) / (edges[2:]-edges[1:-1])[:,None]
    return np.maximum(0,np.minimum(up,down)).astype(np.float32)

MEL = mel_bank()
HANN = np.hanning(WIN).astype(np.float32)

def logmel(audio):
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(audio,((0,0),(WIN//2,WIN//2))),WIN,axis=1)[:,::HOP]
    fft = np.fft.rfft(windows*HANN,n=NFFT,axis=-1)
    power = (fft.real**2+fft.imag**2).astype(np.float32)
    return np.log1p((power @ MEL.T)*10).transpose(0,2,1).copy()

class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.conv = nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.mix = nn.Conv1d(width,width,1)

    def forward(self,x):
        return x + 0.5*self.mix(F.gelu(self.conv(x),approximate='tanh'))

class Student(nn.Module):
    def __init__(self,width,mean,scale,active):
        super().__init__()
        self.encoder = nn.Conv1d(CHANNELS,width,5,padding=2)
        self.blocks = nn.Sequential(*(Block(width,d) for d in (1,2,4,8,16,1,2,4)))
        self.output = nn.Conv1d(width,len(active),1)
        self.register_buffer('mean',torch.from_numpy(mean))
        self.register_buffer('scale',torch.from_numpy(scale))
        self.register_buffer('channel_map',F.one_hot(torch.as_tensor(active),52).float())
        # Map the 10 ms feature timeline onto the teacher's 60 fps sample positions.
        positions = (np.arange(60)*RATE//60).astype(np.float32)/HOP
        self.register_buffer('left',torch.from_numpy(positions.astype(np.int64)))
        self.register_buffer('fraction',torch.from_numpy(positions%1).view(1,1,-1))

    def forward(self,mel):
        encoded = self.blocks(F.gelu(self.encoder(mel),approximate='tanh'))
        lo = torch.index_select(encoded,2,self.left)
        hi = torch.index_select(encoded,2,self.left+1)
        x = self.output(lo+self.fraction*(hi-lo)).transpose(1,2)
        return x*self.scale + self.mean

    def coefficients(self,mel):
        return self(mel).clamp(0,1) @ self.channel_map

class WaveformModel(nn.Module):
    def __init__(self,student):
        super().__init__()
        self.student = student
        phases = 2*np.pi*np.arange(NFFT//2+1)[:,None]*np.arange(WIN)[None,:]/NFFT
        filters = np.concatenate((np.cos(phases),np.sin(phases)))*HANN
        self.register_buffer('filters',torch.from_numpy(filters.astype(np.float32)).unsqueeze(1))
        self.register_buffer('mel',torch.from_numpy(MEL).unsqueeze(2))

    def forward(self,pcm):
        spectrum = F.conv1d(pcm.unsqueeze(1),self.filters,stride=HOP,padding=WIN//2)
        real,imag = spectrum.chunk(2,dim=1)
        mel = (1+F.conv1d(real.square()+imag.square(),self.mel)*10).log()
        return self.student.coefficients(mel)

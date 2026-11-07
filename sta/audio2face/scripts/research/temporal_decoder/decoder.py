"""Causal residual decoder; history contains normalized audio-motion features."""
import torch
from torch import nn
from torch.nn import functional as F
TAPS, FEATURES, WIDTH = 8, 96, 16


class Decoder(nn.Module):
    def __init__(self, center, spread, mode):
        super().__init__()
        self.mode = mode
        self.register_buffer('center', torch.as_tensor(center).float())
        self.register_buffer('spread', torch.as_tensor(spread).float())
        self.input = nn.Linear(TAPS * FEATURES, WIDTH)
        self.output = nn.Linear(WIDTH, 39)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, contexts):
        if self.mode == 'control':
            contexts = contexts[..., -1:, :].expand_as(contexts)
        return self.output(F.gelu(self.input(contexts.flatten(-2)), approximate='tanh'))

    def stream(self, features, history, count):
        normalized = (features - self.center) / self.spread
        joined = torch.cat((history, normalized), dim=1)
        contexts = torch.stack([joined[:, i:i + 30] for i in range(TAPS)], dim=2)
        next_history = joined.index_select(1, count[0] + torch.arange(7, device=features.device))
        return self(contexts), next_history


def encoded(student, mel):
    features = student.blocks(F.gelu(student.encoder(mel), approximate='tanh'))
    lo = torch.index_select(features, 2, student.left)
    hi = torch.index_select(features, 2, student.left + 1)
    return (lo + student.fraction * (hi - lo)).transpose(1, 2)


def waveform_class(base):
    class TemporalWaveform(base.WaveformModel):
        def __init__(self, student, decoder):
            super().__init__(student); self.decoder = decoder

        def forward(self, pcm, history, first, count):
            spectrum = F.conv1d(pcm.unsqueeze(1), self.filters, stride=160, padding=200)
            real, imag = spectrum.chunk(2, dim=1)
            mel = (1 + F.conv1d(real.square() + imag.square(), self.mel) * 10).log()
            features = encoded(self.student, mel)
            indices = (first[0] + torch.arange(30, device=pcm.device)).clamp(max=59)
            selected = features.index_select(1, indices)
            baseline = (self.student.output(selected.transpose(1, 2)).transpose(1, 2) * self.student.scale + self.student.mean).clamp(0, 1)
            delta, next_history = self.decoder.stream(selected, history, count)
            prediction = (baseline + delta * self.student.scale).clamp(0, 1) @ self.student.channel_map
            return prediction, next_history
    return TemporalWaveform

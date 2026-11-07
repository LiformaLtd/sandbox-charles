"""Preserve the old function while adding independent, trainable features."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def initialize_student(student, values):
    old_width = len(values['student.encoder.bias'])
    width = student.encoder.out_channels
    if width < old_width:
        raise ValueError('This experiment only preserves or expands width')
    state = student.state_dict()
    for name, target in state.items():
        source = torch.from_numpy(values['student.' + name].copy())
        if target.shape == source.shape:
            target.copy_(source)
        elif name in ['encoder.weight', 'encoder.bias']:
            target[:old_width].copy_(source)
        elif name == 'output.weight':
            target.zero_()
            target[:, :old_width].copy_(source)
        elif name.endswith('.weight') and name.startswith('blocks.'):
            # Original outputs ignore new inputs at initialization. New rows
            # retain independent PyTorch initialization, never channel copies.
            target[:old_width, :old_width].copy_(source)
            target[:old_width, old_width:].zero_()
        elif name.endswith('.bias') and name.startswith('blocks.'):
            target[:old_width].copy_(source)
        else:
            raise ValueError('Unexpected expanded tensor: ' + name)
    student.load_state_dict(state)
    cosine = 0.
    if width > old_width:
        original = F.normalize(state['encoder.weight'][:old_width].flatten(1), dim=1)
        extra = F.normalize(state['encoder.weight'][old_width:].flatten(1), dim=1)
        cosine = float((extra @ original.T).abs().max())
        if cosine >= .5:
            raise ValueError('New encoder channels are unexpectedly similar to originals')
    return dict(original_width=old_width, width=width, independent_extra_channels=width - old_width,
                max_absolute_new_old_filter_cosine=cosine,
                policy='Random extra features; zero initial feedback into original channels/output; all weights trainable')

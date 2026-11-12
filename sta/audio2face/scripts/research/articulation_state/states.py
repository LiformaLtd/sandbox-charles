"""Geometry-derived states: no text/aligner input, no runtime head."""
import torch
from torch.nn import functional as F


def articulation_targets(gaps, valid):
    # Teacher geometry is in centimetres, at 60 fps. Require both neighbours
    # so padded samples and independent block edges cannot invent transitions.
    level = gaps.median(dim=-1).values
    speed = torch.zeros_like(level)
    speed[:, 1:-1] = (level[:, 2:] - level[:, :-2]) * .5
    mask = torch.zeros_like(valid)
    mask[:, 1:-1] = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    result = torch.zeros_like(level, dtype=torch.long)  # open/approximately steady
    result[speed < -.025] = 1  # closing, > .25 mm per frame
    result[speed > .025] = 2   # opening
    result[level <= .1] = 3    # contact/hold, same 1 mm threshold as geometry metric
    return result.masked_fill(~mask, -100)


def state_loss(logits, targets):
    valid = targets != -100
    if not valid.any():
        return logits.sum() * 0
    weights = logits.new_tensor([1., 2., 2., 2.])
    return F.cross_entropy(logits[valid], targets[valid], weight=weights)

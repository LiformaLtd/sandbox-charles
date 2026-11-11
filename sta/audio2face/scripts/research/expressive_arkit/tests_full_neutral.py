"""Contracts for absolute neutral supervision and the slow brow decoder."""
import unittest
import numpy as np
import torch
from full_neutral_model import NaturalBrows, NaturalFace, conditions
from common import OUT
from gaze import NEW

class NeutralContracts(unittest.TestCase):
    def test_neutral_is_an_explicit_condition(self):
        e=torch.zeros(1,60,10)
        self.assertTrue(torch.equal(conditions(e)[...,0],torch.ones(1,60)))
        e[...,2]=.4
        self.assertTrue(torch.allclose(conditions(e)[...,0],torch.full((1,60),.6)))
        self.assertTrue(torch.allclose(conditions(e).sum(-1),torch.ones(1,60)))

    def test_neutral_brows_use_teacher_statistics(self):
        means=np.ones((11,5),np.float32)*.3
        net=NaturalBrows(np.zeros(96,np.float32),np.ones(96,np.float32),means,np.ones((11,5),np.float32)*.1)
        p=net(torch.zeros(1,60,96),torch.zeros(1,60,10))
        self.assertTrue(torch.allclose(p,torch.full((1,60,5),.3)))
        p.sum().backward()
        self.assertGreater(float(net.output.bias.grad.abs().min()),0)

    def test_neutral_allows_mouth_and_cheek_corrections(self):
        adapter=torch.load(OUT/'adapter.pt',map_location='cpu',weights_only=True)
        gaze=torch.load(NEW/'gaze.pt',map_location='cpu',weights_only=True)
        net=NaturalFace(adapter,gaze,np.zeros((11,5),np.float32),np.ones((11,5),np.float32))
        with torch.no_grad():
            net.fast.output.bias[14]=.2
            net.fast.output.bias[46]=.3
        p=net(torch.zeros(1,60,96),torch.zeros(1,60,52),torch.zeros(1,60,10),torch.zeros(1,60,3))
        self.assertTrue(torch.allclose(p[:,:,14],torch.full((1,60),.2)))
        self.assertTrue(torch.allclose(p[:,:,46],torch.full((1,60),.3)))

if __name__=='__main__':unittest.main()

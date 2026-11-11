import unittest
import numpy as np
import torch
from brow_model import BrowDecoder


class BrowContracts(unittest.TestCase):
    def net(self):
        weights=torch.zeros(5,64,1);weights[:,0,0]=.05
        return BrowDecoder(weights,torch.ones(5)*.3)

    def test_neutral_is_learned_and_moves(self):
        net=self.net();latent=torch.zeros(1,64,60);latent[:,0]=torch.sin(torch.arange(60)*2*np.pi/60)
        prediction=net(latent,torch.zeros(1,60,10))
        self.assertGreater(float(prediction.max()-prediction.min()),.05)
        (prediction-.5).square().mean().backward()
        self.assertTrue(torch.all(net.projection.bias.grad.abs()>0))

    def test_half_second_anchor_grid(self):
        net=self.net();latent=torch.randn(1,64,90)
        a=net.slow_latents(latent[:,:,:60]);b=net.slow_latents(latent[:,:,30:90])
        self.assertTrue(torch.equal(a[:,:,3:6],b[:,:,:3]))

    def test_fast_signal_attenuated_without_freezing_gestures(self):
        net=self.net();strength=[]
        for hz in [1,10]:
            latent=torch.zeros(1,64,60);latent[:,0]=torch.sin(torch.arange(60)*2*np.pi*hz/60)
            with torch.no_grad():p=net(latent,torch.zeros(1,60,10))[0,:,0].numpy()
            strength.append(float(p.std()))
        self.assertGreater(strength[0],.025)
        self.assertLess(strength[1],.3*strength[0])


if __name__=='__main__':unittest.main()

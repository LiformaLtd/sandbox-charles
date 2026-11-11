"""Boundary checks for retargeting and the frozen articulation contract."""
import unittest
import numpy as np
import torch
from fit import gaze_coefficients, bounded_solve
from model import ExpressionAdapter


class Contracts(unittest.TestCase):
    def test_gaze_signed_axes_and_exclusion(self):
        right=np.array([[25,25,25,-25],[-25,-25,-25,25]],np.float32)
        weights=gaze_coefficients(right)
        self.assertTrue(np.array_equal(weights[0,[1,2,8,9]],np.ones(4)))
        self.assertTrue(np.array_equal(weights[1,[3,4,10,11]],np.ones(4)))
        for left,right in [(1,4),(2,3),(8,11),(9,10)]:
            self.assertFalse(np.any(weights[:,left]*weights[:,right]))

    def test_gaze_clip_and_invalid_values(self):
        self.assertEqual(float(gaze_coefficients(np.full((2,4),100,np.float32)).max()),1.)
        for invalid in [np.zeros((2,5)),np.array([[np.nan,0,0,0]])]:
            with self.assertRaises(ValueError):gaze_coefficients(invalid)

    def test_bounded_fit_obeys_coupled_objective(self):
        gram=np.array([[2.,.5],[.5,3.]],np.float32)
        truth=np.array([[.3,.7],[0.,1.]],np.float32)
        actual=bounded_solve(np.einsum('ik,kj->ij',truth,gram,optimize=False),gram)
        np.testing.assert_allclose(actual,truth,atol=1e-5)

    def net(self):
        protected=np.array([i for i in range(52) if i not in [1,2,3,4,8,9,10,11,51]])
        return ExpressionAdapter(protected,np.zeros(96,np.float32),np.ones(96,np.float32)),protected

    def test_neutral_protection_with_trained_nonzero_adapter(self):
        net,protected=self.net()
        with torch.no_grad():net.output.weight.normal_(0,.1);net.output.bias.fill_(.2)
        speech=torch.rand((2,60,52))
        out=net(torch.randn(2,60,96),speech,torch.zeros(2,60,10))
        self.assertTrue(torch.equal(out[:,:,protected],speech[:,:,protected]))

    def test_all_52_output_heads_receive_gradient(self):
        net,_=self.net()
        out=net(torch.randn(2,60,96),torch.zeros(2,60,52),torch.ones(2,60,10)*.1)
        (out-.5).square().mean().backward()
        self.assertTrue(torch.all(net.output.bias.grad.abs()>0))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()

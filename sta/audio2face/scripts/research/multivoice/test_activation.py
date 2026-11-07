"""The stable shader activation must preserve the trained float32 GELU."""
import unittest
import torch
from torch.nn import functional as F


class Activation(unittest.TestCase):
    def test_saturation_bound_matches_independent_pytorch_gelu(self):
        # Includes the observed encoder range (-40..7) and last-block peaks
        # above 13 that overflow unbounded WebKit/Metal tanh intermediates.
        x = torch.linspace(-64, 64, 200001, dtype=torch.float32)
        argument = .7978845608 * (x + .044715 * x * x * x)
        bounded = .5 * x * (1 + torch.tanh(argument.clamp(-10, 10)))
        reference = F.gelu(x, approximate='tanh')
        torch.testing.assert_close(bounded, reference, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(bounded).all())


if __name__ == '__main__':
    unittest.main()

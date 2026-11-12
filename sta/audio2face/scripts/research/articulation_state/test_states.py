import unittest
import torch
from states import articulation_targets, state_loss


class StateTests(unittest.TestCase):
    def test_contact_and_direction_are_geometry_timed(self):
        level = torch.tensor([[.6, .6, .4, .2, .05, .05, .2, .4, .6, .6]])
        labels = articulation_targets(level[..., None].repeat(1, 1, 9), torch.ones_like(level, dtype=torch.bool))
        self.assertEqual(labels.tolist(), [[-100, 1, 1, 1, 3, 3, 2, 2, 2, -100]])

    def test_padding_cannot_create_transition_or_gradient(self):
        valid = torch.tensor([[False, False, True, True, True, True, False]])
        level = torch.tensor([[0., 0., .5, .5, .5, .5, 0.]])
        labels = articulation_targets(level[..., None].repeat(1, 1, 9), valid)
        self.assertEqual(labels.tolist(), [[-100, -100, -100, 0, 0, -100, -100]])
        logits = torch.zeros(1, 7, 4, requires_grad=True)
        loss = state_loss(logits, labels); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad[labels == -100].abs().sum()), 0.)

    def test_all_padding_is_finite(self):
        x = torch.zeros(1, 3, 4, requires_grad=True)
        loss = state_loss(x, torch.full((1, 3), -100));loss.backward()
        self.assertEqual(float(loss), 0.)


if __name__ == '__main__':
    unittest.main()

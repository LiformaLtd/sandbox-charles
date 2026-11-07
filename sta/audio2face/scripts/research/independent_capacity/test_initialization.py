from pathlib import Path
import importlib.util
import os
import sys
import unittest
import numpy as np
import torch
from initialization import initialize_student

if os.environ.get('A2F_ONNX_SITE_PACKAGES'):
    sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper

APP = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('capacity_fixture_model', APP / '.local/compact-coverage-v2/base/sources/model.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)


class IndependentFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        graph = onnx.load(APP / '.local/compact-coverage-v2/coverage/runs/control-replay-s37/model.onnx')
        cls.values = {t.name: numpy_helper.to_array(t).copy() for t in graph.graph.initializer}
        cls.active = cls.values['student.channel_map'].argmax(1)
        torch.set_num_threads(4)

    def make(self, width):
        torch.manual_seed(37)
        student = base.Student(width, self.values['student.mean'], self.values['student.scale'], self.active)
        audit = initialize_student(student, self.values)
        return student, audit

    def test_expansion_preserves_v4_function_without_duplicate_filters(self):
        original, _ = self.make(96); expanded, audit = self.make(128)
        x = torch.linspace(0, 8, 3 * 64 * 101).reshape(3, 64, 101)
        with torch.no_grad():
            torch.testing.assert_close(expanded(x), original(x), atol=2e-6, rtol=2e-6)
        self.assertEqual(audit['independent_extra_channels'], 32)
        self.assertLess(audit['max_absolute_new_old_filter_cosine'], .5)
        self.assertEqual(sum(p.numel() for p in original.parameters()), 331047)
        self.assertEqual(sum(p.numel() for p in expanded.parameters()), 572455)

    def test_new_features_receive_gradients_after_output_opens(self):
        student, _ = self.make(128)
        x = torch.rand(2, 64, 101)
        target = torch.rand(2, 60, 39) * .5
        optimizer = torch.optim.SGD(student.parameters(), lr=.01)
        loss = (student(x) - target).square().mean(); loss.backward()
        self.assertGreater(float(student.output.weight.grad[:, 96:].abs().sum()), 0)
        optimizer.step(); optimizer.zero_grad()
        (student(x) - target).square().mean().backward()
        self.assertGreater(float(student.encoder.weight.grad[96:].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()

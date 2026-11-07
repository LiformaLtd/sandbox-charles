"""Protect the deployment contract and training-only supervision paths."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import train


class Contract(unittest.TestCase):
    def make(self):
        torch.manual_seed(37)
        s = train.base.Student(96, np.zeros(39, np.float32), np.ones(39, np.float32), np.arange(39))
        return train.TrainingModel(s, 39)

    def test_auxiliary_losses_reach_existing_encoder(self):
        for index in [1, 2]:
            net = self.make()
            outputs = net(torch.randn(2, 64, 101))
            if index == 1:
                loss = F.cross_entropy(outputs[1].reshape(-1, 39), torch.randint(39, (60,)))
            else:
                loss = F.smooth_l1_loss(outputs[2], torch.randn(2, 30, 768))
            loss.backward()
            grad = net.student.encoder.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)

    def test_training_head_preserves_student_prediction(self):
        net = self.make().eval()
        mel = torch.randn(2, 64, 101)
        torch.testing.assert_close(net(mel)[0], net.student(mel)[:, 15:45], atol=1e-6, rtol=1e-6)

    def test_export_has_only_original_network(self):
        net = self.make().eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.onnx'
            train.torch.onnx.export(train.base.WaveformModel(net.student), torch.zeros(1, 16000), str(path),
                                    input_names=['pcm'], output_names=['blendshapes'], opset_version=17, dynamo=False)
            graph = train.onnx.load(path)
            train.onnx.checker.check_model(graph)
            self.assertFalse(any('head' in t.name for t in graph.graph.initializer))
            self.assertEqual(sum(p.numel() for p in net.student.parameters()), 331047)
            self.assertLessEqual(path.stat().st_size, 2260000)

    def test_partial_teacher_crop_uses_exact_sample_grid(self):
        for length in [4101, 8000, 12345, 51317]:
            selected = []
            offsets = np.arange(15, 45) * 16000 // 60
            for start in range(-16000, length - 4000, 8000):
                indices = start + offsets
                valid = (indices >= 0) & (indices < length)
                inverse = ((indices[valid] - start) * 60 + 15999) // 16000
                np.testing.assert_array_equal(inverse, np.arange(15, 45)[valid])
                selected.extend(indices[valid])
            np.testing.assert_array_equal(selected, np.arange((length * 60 + 15999) // 16000) * 16000 // 60)


if __name__ == '__main__':
    unittest.main()

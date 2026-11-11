import unittest
from pathlib import Path
import numpy as np
import torch
from gaze_model import GazeHead
from infer_full import sequence


class FullFaceContracts(unittest.TestCase):
    def head(self):return GazeHead(np.zeros(96,np.float32),np.ones(96,np.float32))

    def test_semantic_gaze_axes(self):
        h=self.head();w=h.coefficients(torch.tensor([[[-.3,.4,.2,-.1]]]))[0,0]
        self.assertAlmostEqual(float(w[11]),.3,places=6)
        self.assertAlmostEqual(float(w[9]),.4,places=6)
        self.assertAlmostEqual(float(w[1]),.2,places=6)
        self.assertAlmostEqual(float(w[2]),.1,places=6)
        self.assertEqual(float(w[[8,10,4,3]].sum()),0.)

    def test_opposed_gaze_is_exclusive_and_bounded(self):
        w=self.head().coefficients(torch.randn(3,60,4)*2)
        for a,b in [(1,4),(2,3),(8,11),(9,10)]:self.assertFalse(torch.any(w[:,:,a]*w[:,:,b]))
        self.assertLessEqual(float(w.max()),1.)
        self.assertGreaterEqual(float(w.min()),0.)

    def test_audio_rejection_precedes_model_loading(self):
        for bad in [np.zeros(0),np.array([np.nan]),np.ones((2,2)),np.array([2.])]:
            with self.assertRaises(ValueError):sequence(Path('absent.onnx'),bad)

    def test_invalid_controls_rejected(self):
        for emotion,intensity,style in [('neutral',-1.,'James'),('unknown',1.,'James'),('neutral',1.,'unknown')]:
            with self.assertRaises(ValueError):sequence(Path('absent.onnx'),np.zeros(16000),emotion,intensity,style)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()

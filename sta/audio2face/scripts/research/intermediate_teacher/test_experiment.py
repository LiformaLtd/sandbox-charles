"""Boundary contracts: independent features, unchanged initial predictions and export layout."""
from common import *
import unittest,numpy as np,onnx
from onnx import numpy_helper
import mlx.core as mx,mlx.nn as nn
from mlx.utils import tree_flatten
from model import Predictor,Learner,feature_loss
from mlx_backend import Student

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values={t.name:numpy_helper.to_array(t).copy() for t in onnx.load(BASELINE).graph.initializer}
        cls.x=mx.array(np.random.default_rng(37).normal(size=(2,64,101)).astype(np.float32))
    def test_widening_preserves_output_but_adds_independent_features(self):
        mx.random.seed(37);small=Predictor(self.values);wide=Predictor(self.values,192)
        np.testing.assert_allclose(np.array(wide(self.x)),np.array(small(self.x)),atol=1e-6,rtol=1e-5)
        features=np.array(wide.hidden(self.x)[-1])
        self.assertGreater(float(np.std(features[:,:,96:])),.01)
        self.assertGreater(float(np.sqrt(np.mean((features[:,:,:96]-features[:,:,96:])**2))),.01)
        weights=wide.torch_weights()
        self.assertEqual(weights['encoder.weight'].shape,(192,64,5))
        for key in weights:
            if key.startswith('blocks.') and key.endswith('weight'):
                self.assertEqual(float(np.abs(weights[key][:96,96:]).max()),0.)
    def test_existing_backend_parity(self):
        expected=Student(self.values);actual=Predictor(self.values)
        np.testing.assert_allclose(np.array(actual(self.x)),np.array(expected(self.x)),atol=1e-6,rtol=1e-5)
        self.assertEqual(sum(v.size for _,v in tree_flatten(actual.trainable_parameters())),331047)
    def test_hint_mask_and_frozen_target(self):
        p=mx.array([[[1.,0.],[0.,1.]]]);t=mx.array([[[1.,0.],[-1.,0.]]]);valid=mx.array([[True,False]])
        self.assertLess(float(feature_loss(p,t,valid)),1e-5)
        grad=mx.grad(lambda target:feature_loss(p,target,mx.array([[True,True]])))(t)
        self.assertEqual(float(mx.abs(grad).max()),0.)
        dp=mx.grad(lambda pred:feature_loss(pred,t,mx.array([[True,True]])))(p)
        self.assertGreater(float(mx.abs(dp).max()),.1)
    def test_student_initialization_matches_between_arms(self):
        mx.random.seed(73);a=Learner(self.values)
        mx.random.seed(73);b=Learner(self.values)
        for (ka,va),(kb,vb) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):
            self.assertEqual(ka,kb);np.testing.assert_array_equal(np.array(va),np.array(vb))
    def test_added_channels_receive_gradient(self):
        mx.random.seed(37);wide=Predictor(self.values,192)
        _,gradient=nn.value_and_grad(wide,lambda n:mx.mean(n(self.x)**2))(wide)
        self.assertGreater(float(mx.abs(gradient['output']['weight'][:,:,96:]).max()),1e-5)
if __name__=='__main__':unittest.main()

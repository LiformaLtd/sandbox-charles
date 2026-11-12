"""Padding and amplitude semantics of the training-only hint objective."""
import unittest,numpy as np
from model import cosine_loss,mx
class HintTests(unittest.TestCase):
 def test_padding_cannot_change_loss_or_gradients(self):
  p=mx.array(np.random.default_rng(37).normal(size=(2,3,5)).astype(np.float32));t=p*.3;mask=mx.array([[True,False,True],[False,True,False]])
  a=cosine_loss(p,t,mask);changed=mx.where(mask[...,None],t,t+10000)
  b=cosine_loss(p,changed,mask);gradient=mx.grad(lambda x:cosine_loss(x,changed,mask))(p)
  self.assertAlmostEqual(float(a),float(b),places=7);self.assertTrue(np.all(np.array(gradient)[~np.array(mask)]==0))
 def test_feature_scale_does_not_change_direction_loss(self):
  p=mx.array([[[2.,-3.,4.]]]);t=mx.array([[[1.,5.,-2.]]]);m=mx.array([[True]])
  self.assertAlmostEqual(float(cosine_loss(p,t,m)),float(cosine_loss(p*30,t*.4,m)),places=5)
 def test_no_valid_frames_are_finite_and_zero(self):
  z=mx.zeros((1,2,768));m=mx.array([[False,False]])
  self.assertEqual(float(cosine_loss(z,z,m)),0.);self.assertTrue(np.isfinite(np.array(mx.grad(lambda p:cosine_loss(p,z,m))(z))).all())
if __name__=='__main__':unittest.main()

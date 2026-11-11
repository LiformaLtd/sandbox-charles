from common import *
import unittest,numpy as np,onnx
from onnx import numpy_helper
import mlx.core as mx,mlx.nn as nn
from mlx_backend import Student
from model import Frontend,Model
from data import Dataset

class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer}
  cls.batch=Dataset(['old-train']).batch(np.arange(3));cls.power=mx.array(cls.batch['power'])
 def test_initial_frontend_matches_original_features(self):
  from features import Dataset as Original
  reference=Original(['old-train']).batch(np.arange(3))['mel']
  for learn in [False,True]:np.testing.assert_allclose(np.array(Frontend(self.values['mel'][:,:,0],learn)(self.power)),reference,atol=2e-6,rtol=2e-6)
 def test_initial_predictions_match(self):
  a,b=Model(self.values,False),Model(self.values,True)
  np.testing.assert_array_equal(np.array(a(self.power)),np.array(b(self.power)))
  from features import Dataset as Original
  expected=np.array(Student(self.values)(mx.array(Original(['old-train']).batch(np.arange(3))['mel'])))
  np.testing.assert_allclose(np.array(a(self.power)),expected,atol=1e-5,rtol=1e-5)
 def test_bank_gradient_only_in_learned_arm(self):
  norms=[]
  for learn in [False,True]:
   f=Frontend(self.values['mel'][:,:,0],learn)
   _,g=nn.value_and_grad(f,lambda n:mx.mean(n(self.power)))(f)
   norms.append(float(mx.abs(g['weights']).max()))
  self.assertEqual(norms[0],0.);self.assertGreater(norms[1],1e-8)
 def test_nonnegative_bank_and_finite_log(self):
  f=Frontend(self.values['mel'][:,:,0],True);f.weights=mx.full(f.weights.shape,-1000.);f.project()
  self.assertGreaterEqual(float(mx.min(f.bank())),0.)
  self.assertTrue(np.isfinite(np.array(f(self.power))).all())
  np.testing.assert_array_equal(np.array(f(mx.zeros_like(self.power))),0.)
 def test_original_constants_are_frozen(self):
  f=Frontend(self.values['mel'][:,:,0],True)
  self.assertEqual(set(f.trainable_parameters()),{'weights'})
  self.assertEqual(int(f.weights.size),16448)
 def test_source_batch_alignment(self):
  from features import Dataset as Original
  original=Original(['old-train']).batch([2,0,2]);actual=Dataset(['old-train']).batch([2,0,2])
  for k in ['weights','gaps','valid']:np.testing.assert_array_equal(original[k],actual[k])
 def test_animation_loss_reaches_bank_without_initial_student_gradient_change(self):
  from model import loss
  from mlx.utils import tree_flatten
  rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active']
  names=json.loads((WORK/'arkit-student/channels.json').read_text())
  args=[mx.array(self.batch[k]) for k in ['power','weights','gaps','valid']]+[mx.array(rig['gap_basis'][active]),mx.array(rig['rest']+rig['gap_offset']),mx.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active])]
  gradients=[]
  for learn in [False,True]:
   net=Model(self.values,learn);_,g=nn.value_and_grad(net,loss)(net,*args);gradients.append(g)
  self.assertEqual(float(mx.abs(gradients[0]['frontend']['weights']).max()),0.)
  self.assertGreater(float(mx.abs(gradients[1]['frontend']['weights']).max()),1e-8)
  for (ka,a),(kb,b) in zip(tree_flatten(gradients[0]['student']),tree_flatten(gradients[1]['student'])):
   self.assertEqual(ka,kb);np.testing.assert_array_equal(np.array(a),np.array(b))
 def test_zero_support_weights_can_receive_gradient_and_activate(self):
  import mlx.optimizers as optim
  f=Frontend(self.values['mel'][:,:,0],True)
  _,g=nn.value_and_grad(f,lambda n:-mx.mean(n(self.power)))(f)
  zero=self.values['mel'][:,:,0]==0
  self.assertGreater(float(np.abs(np.array(g['weights'])[zero]).max()),1e-6)
  optimizer=optim.AdamW(learning_rate=1e-6,weight_decay=0.)
  optimizer.update(f,g);f.project();mx.eval(f.parameters())
  self.assertTrue((np.array(f.bank())[zero]>0).any())
if __name__=='__main__':unittest.main()

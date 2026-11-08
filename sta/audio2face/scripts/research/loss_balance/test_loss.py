import unittest,numpy as np,mlx.core as mx,mlx.nn as nn
from mlx.utils import tree_flatten
from common import *
from features import Dataset
from mlx_backend import Student,objective as original
from loss import objective,components
import onnx
from onnx import numpy_helper
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.net=Student({v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer});b=Dataset(['old-train']).batch(np.arange(8));r=np.load(WORK/'arkit-student/rig-metrics.npz');names=json.loads((WORK/'arkit-student/channels.json').read_text());cls.args=[mx.array(b[k]) for k in ['mel','weights','gaps','valid']]+[mx.array(r['gap_basis'][r['active']]),mx.array(r['rest']+r['gap_offset']),mx.array([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in r['active']])]
 def test_control_value_and_gradient(self):
  a,ga=nn.value_and_grad(self.net,original)(self.net,*self.args);b,gb=nn.value_and_grad(self.net,objective)(self.net,*self.args,1.)
  self.assertEqual(float(a),float(b))
  for (_,x),(_,y) in zip(tree_flatten(ga),tree_flatten(gb)):np.testing.assert_array_equal(np.array(x),np.array(y))
 def test_only_coefficient_weight_changes(self):
  a=objective(self.net,*self.args,1.);b=objective(self.net,*self.args,4.);shape=components(self.net,*self.args)[0];self.assertAlmostEqual(float(b-a),float(3*shape),places=6)
if __name__=='__main__':unittest.main()

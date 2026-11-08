"""Coordinate-frame and dense-objective invariants on the actual James rig."""
import unittest,numpy as np
from geometry import prepare,statistics,squared_error
class GeometryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rig=prepare()
 def test_quadratic_keeps_dense_error_including_unrepresentable_residual(self):
  r=self.rig;rng=np.random.default_rng(37);raw=rng.normal(0,.03,(3,72006)).astype(np.float32);weights=rng.uniform(0,1,(3,39)).astype(np.float32)
  target=raw.reshape(3,24002,3)[:,r['ids']]+r['offset'];pred=np.einsum('tc,cva->tva',weights,r['basis'],optimize=False);dense=np.square(pred-target).mean(1)
  quadratic=squared_error(weights,statistics(raw,r),r['gram']);np.testing.assert_allclose(quadratic,dense,rtol=2e-5,atol=2e-5)
 def test_matching_rig_neutral_has_zero_error(self):
  r=self.rig;raw=np.zeros((1,24002,3),np.float32);raw[:,r['ids']]=-r['offset'];stats=statistics(raw.reshape(1,72006),r);error=squared_error(np.zeros((1,39),np.float32),stats,r['gram']);np.testing.assert_allclose(error,0,atol=1e-12)
 def test_all_gap_landmarks_are_in_dense_region(self):
  from geometry import PAIRS
  self.assertTrue(set(PAIRS.ravel())<=set(self.rig['ids'].tolist()))
if __name__=='__main__':unittest.main()

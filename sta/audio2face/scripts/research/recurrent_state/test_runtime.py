"""Frame causality, cross-block carry and immutable experiment checks."""
import unittest,numpy as np,mlx.core as mx
from common import *
from model import Model
from mlx.utils import tree_unflatten
import onnx
from onnx import numpy_helper

class RuntimeChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};cls.model=Model(values)
  with np.load(OUT/'recurrent/best.npz') as z:cls.model.update(tree_unflatten([(k,mx.array(z[k])) for k in z.files]))
  cls.x=mx.array(np.random.default_rng(8).normal(size=(1,4,64,101)).astype(np.float32));v=np.ones((1,120),bool);v[:,:15]=False;v[:,113:]=False;cls.valid=mx.array(v)
 def test_future_blocks_cannot_change_earlier_output(self):
  x=self.x;y=mx.concatenate((x[:,:3],x[:,3:]+4),1);a,_=self.model(x,self.valid,mx.zeros((1,32)));b,_=self.model(y,self.valid,mx.zeros((1,32)));np.testing.assert_allclose(np.array(a[:,:90]),np.array(b[:,:90]),atol=1e-6,rtol=1e-6)
 def test_chunk_partition_preserves_state(self):
  full,state=self.model(self.x,self.valid,mx.zeros((1,32)));a,h=self.model(self.x[:,:2],self.valid[:,:60],mx.zeros((1,32)));b,h=self.model(self.x[:,2:],self.valid[:,60:],h)
  np.testing.assert_allclose(np.array(full),np.array(mx.concatenate((a,b),1)),atol=1e-6,rtol=1e-6);np.testing.assert_allclose(np.array(state),np.array(h),atol=1e-6,rtol=1e-6)
 def test_schedule_continuity_and_resets(self):
  z=np.load(OUT/'schedule.npz');ix=z['indices'];reset=z['resets'];self.assertTrue(reset[0].all())
  for t in range(1,len(ix)):
   for j in range(20):
    if not reset[t,j]:self.assertEqual(ix[t,j,0],ix[t-1,j][ix[t-1,j]>=0][-1]+1)
 def test_all_frozen_hashes(self):
  protocol=json.loads((OUT/'protocol.json').read_text());self.assertEqual(sha(BASELINE),EXPECTED);self.assertEqual(sha(OUT/'schedule.npz'),protocol['schedule_sha256'])
  for name,digest in protocol['sources'].items():self.assertEqual(sha(APP/name),digest,name)
  for name,digest in protocol['arrays'].items():self.assertEqual(sha(name),digest,name)
  for name,digest in json.loads((OUT/'evaluation-lock.json').read_text())['sources'].items():self.assertEqual(sha(APP/name),digest,name)
  for arm in ['control','recurrent']:
   report=json.loads((OUT/arm/'selection.json').read_text());self.assertEqual(sha(OUT/arm/'best.npz'),report['best_sha256']);self.assertEqual(sha(OUT/arm/'model.onnx'),report['model_sha256']);self.assertEqual(report['parameters'],344750)
if __name__=='__main__':unittest.main()

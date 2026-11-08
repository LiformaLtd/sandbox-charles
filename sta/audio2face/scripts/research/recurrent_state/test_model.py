import unittest,numpy as np,mlx.core as mx
from common import *
import onnx
from onnx import numpy_helper
from model import Model
from data import clips,TRAIN,batch,Dataset
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer}
 def test_zero_head_is_v4(self):
  b=batch(Dataset(TRAIN),np.array([[0,1,2,3]]));x=mx.array(b['mel']);valid=mx.array(b['valid'])
  for mode in (False,True):
   m=Model(self.values,mode);actual,_=m(x,valid,mx.zeros((1,32)));expected=m.student(x.reshape(4,64,101))[:,15:45].reshape(1,120,39);np.testing.assert_array_equal(np.array(actual),np.array(expected))
 def test_padding_does_not_advance_state(self):
  m=Model(self.values);h=mx.ones((1,32));_,state=m(mx.zeros((1,4,64,101)),mx.zeros((1,120),mx.bool_),h);np.testing.assert_array_equal(np.array(state),np.array(h))
 def test_control_ignores_input_state_on_valid_frames(self):
  m=Model(self.values,False);m.head.weight=mx.ones_like(m.head.weight)*.1;x=mx.zeros((1,1,64,101));v=mx.ones((1,30),mx.bool_)
  a,_=m(x,v,mx.ones((1,32)));b,_=m(x,v,mx.zeros((1,32)));np.testing.assert_array_equal(np.array(a),np.array(b))
 def test_schedule_never_crosses_clips(self):
  z=np.load(OUT/'schedule.npz');rows=clips(TRAIN)
  for ids in z['indices'].reshape(-1,4):
   ids=ids[ids>=0]
   self.assertTrue(np.all(np.diff(ids)==1))
   self.assertTrue(any(c['start']<=ids[0] and ids[-1]<c['start']+c['count'] for c in rows))
if __name__=='__main__':unittest.main()

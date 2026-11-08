"""Known projection solutions, numerical feasibility, and a real guarded update."""
from common import *
import numpy as np,mlx.core as mx
from protection import project,dot

def main():
 cases=[([1.,1.],[[1.,0.],[0.,1.]],[0.,0.]),([1.,-2.],[[1.,0.],[1.,1.]],[0.,-2.]),([1.,2.],[[1.,0.],[-1.,0.]],[0.,2.]),([-1.,-1.],[[1.,0.],[0.,1.]],[-1.,-1.]),([1.,1.],[[1.,0.],[-.5,1.]],[0.,0.])]
 for d,g,want in cases:
  result,_=project(mx.array(d),list(map(mx.array,g)));np.testing.assert_allclose(np.array(result),want,atol=1e-6)
 rng=np.random.default_rng(555)
 for _ in range(100):
  d=mx.array(rng.normal(size=100).astype(np.float32));guards=[mx.array(rng.normal(size=100).astype(np.float32)) for _ in range(2)];result,info=project(d,guards)
  assert all(dot(g,result)<=tol+1e-6 for g,tol in zip(guards,info['tolerance']));assert dot(result,result)<=dot(d,d)+1e-4
 freeze(OUT/'projection-tests.json',dict(known_solutions=5,random_properties=100,passes=True,source_sha256=sha(__file__),projection_sha256=sha(HERE/'protection.py')))
 print('Projection tests passed.',flush=True)
if __name__=='__main__':main()

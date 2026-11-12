"""Tests for geometry correspondence and development/evaluation isolation."""
import unittest
from pathlib import Path
import numpy as np
from geometry_match import (geometry_features,fit_source_pose,build_geometry_library,
                            DEVELOPMENT_FACES,NEW_TEST_FACES)
from mapping import UP,DOWN


class GeometryMatchTests(unittest.TestCase):
    def test_geometry_features_are_linear_and_distinguish_spread_from_aperture(self):
        a=np.zeros((478,2));a[DOWN,1]=.1
        b=np.zeros((478,2));b[291,0]=.2;b[61,0]=-.2
        np.testing.assert_allclose(geometry_features(a+b),geometry_features(a)+geometry_features(b))
        self.assertGreater(np.linalg.norm(geometry_features(a)-geometry_features(b)),.1)

    def test_geometry_solve_recovers_known_pose_and_preserves_unowned_controls(self):
        basis=np.zeros((478,2,52));basis[DOWN,1,0]=.4
        basis[291,0,1]=.2;basis[61,0,1]=-.2
        reference=np.full(52,.1);expected=reference.copy();expected[:2]=[.7,.4]
        owned=np.zeros(52,bool);owned[:2]=True
        displacement=np.einsum('nck,k->nc',basis,expected-reference)
        w,diagnostics=fit_source_pose(reference,basis,reference,displacement,owned,1e-12)
        np.testing.assert_allclose(w,expected,atol=1e-8)
        np.testing.assert_array_equal(w[~owned],reference[~owned])
        self.assertLess(diagnostics['geometry_rms'],1e-8)

    def test_impossible_geometry_respects_coefficient_bounds(self):
        basis=np.zeros((478,2,52));basis[DOWN,1,0]=.1
        owned=np.zeros(52,bool);owned[0]=True
        desired=np.zeros((478,2));desired[DOWN,1]=10
        w,diagnostics=fit_source_pose(np.zeros(52),basis,np.zeros(52),desired,owned)
        self.assertTrue(np.all((w>=0)&(w<=1)))
        self.assertGreater(diagnostics['geometry_rms'],.1)

    def test_library_does_not_request_test_face_meshes(self):
        calls=[]
        def loader(path):
            calls.append(path.parent.name)
            ref=np.zeros((478,2));ref[61]=[-.5,0];ref[291]=[.5,0]
            pose=ref.copy();pose[DOWN,1]=.1
            return {},{},dict(neutral=ref,aa=pose),None,None
        neutral=np.zeros((478,3));neutral[61,0]=-.5;neutral[291,0]=.5
        basis=np.zeros((478,3,52));basis[DOWN,1,0]=.2
        names=['jawOpen']+['unowned']*51
        build_geometry_library({'aa':dict(coefficients=np.zeros(52),stem='seed',frame=0)},
            neutral,basis,np.zeros(52),Path('/not-read'),loader,{'aa':[]},names)
        self.assertEqual(calls,DEVELOPMENT_FACES)
        self.assertFalse(set(calls)&set(NEW_TEST_FACES))


if __name__=='__main__':unittest.main()

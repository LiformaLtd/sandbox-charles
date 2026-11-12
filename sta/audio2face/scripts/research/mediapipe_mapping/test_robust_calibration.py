import unittest
import numpy as np
from mapping import (UP,DOWN,ANCHORS,INNER_UP,INNER_DOWN,OUTER_UP,OUTER_DOWN,
                     align_to_reference)
from robust_calibration import robust_alignment,prepare_observations,fit_robust_mapping


def reference_face():
    p=np.zeros((478,2))
    for curve,height in [(INNER_UP,-.015),(INNER_DOWN,.015),(OUTER_UP,-.06),(OUTER_DOWN,.06)]:
        t=np.linspace(0,1,len(curve));p[curve,0]=t-.5;p[curve,1]=height*np.sin(t*np.pi)
    return p


class RobustTests(unittest.TestCase):
    def test_isolated_contour_error_is_repaired_without_mutating_input(self):
        ref=reference_face();clean=ref.copy();clean[DOWN,1]+=.1
        noisy=clean.copy();noisy[13,1]+=.25;before=noisy.copy()
        poses,confidence,notes=prepare_observations(dict(neutral=ref,E=noisy))
        self.assertIn(13,notes['E']['isolated_repaired_vertices'])
        self.assertLess(np.linalg.norm(poses['E'][13]-clean[13]),.005)
        self.assertLess(confidence['E'][13],1)
        np.testing.assert_array_equal(noisy,before)

    def test_closed_pose_is_corrected_while_parted_neutral_is_preserved(self):
        ref=reference_face();bad=ref.copy();bad[DOWN,1]+=.04;bad[UP,1]-=.04
        closed,_,notes=prepare_observations(dict(neutral=ref,PP=bad))
        np.testing.assert_array_equal(closed['neutral'],ref)
        gap=closed['PP'][DOWN,1]-closed['PP'][UP,1]
        self.assertLess(gap.max(),.011)
        self.assertLessEqual(notes['PP']['maximum_closed_pose_repair'],.05)

    def test_bad_alignment_anchor_does_not_move_the_whole_mouth(self):
        rng=np.random.default_rng(20);ref=rng.normal(size=(478,2))*.3
        ref[61]=[-.5,0];ref[291]=[.5,0]
        p=ref*1.2+[.1,.2];p[ANCHORS[0]]+=[.5,-.3]
        robust=robust_alignment(p,ref);ordinary=align_to_reference(p,ref)
        self.assertLess(np.linalg.norm(robust[13]-ref[13]),.002)
        self.assertGreater(np.linalg.norm(ordinary[13]-ref[13]),.005)

    def test_robust_map_preserves_nonzero_reference_exactly(self):
        rng=np.random.default_rng(3);ref=reference_face();prior=np.zeros((478,2,52))
        reference=np.full(52,.1);coeff=reference+rng.uniform(0,.1,size=(5,52))
        target=np.repeat(ref[None],5,axis=0);target[0,13,1]=.5
        m,stats=fit_robust_mapping(ref,target,coeff,reference,prior)
        np.testing.assert_array_equal(ref+np.einsum('nck,k->nc',m,reference-reference),ref)
        self.assertTrue(np.isfinite(m).all())


if __name__=='__main__':unittest.main()

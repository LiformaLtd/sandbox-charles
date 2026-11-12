"""Geometry/solver invariants that prevent misleading feasibility measurements."""
import unittest
import numpy as np
from mapping import (fit_mapping,evaluate,contact_correct,gaps,UP,DOWN,
                     local_frame,normalize,denormalize,align_to_reference,ANCHORS)
from run_probe import infer_windows


class MappingTests(unittest.TestCase):
    def test_nonzero_reference_is_exact_and_unsupported_directions_keep_prior(self):
        rng=np.random.default_rng(4)
        ref=rng.normal(size=(478,2));prior=rng.normal(size=(478,2,52))*.01
        reference_weights=np.full(52,.2)
        q=np.zeros((3,52));q[:3,:3]=np.eye(3)
        target=np.array([ref+prior[:,:,i]+.02 for i in range(3)])
        m,stats=fit_mapping(ref,target,reference_weights+q,reference_weights,prior,.001)
        np.testing.assert_array_equal(evaluate(ref,m,reference_weights,reference_weights),ref)
        np.testing.assert_array_equal(m[:,:,3:],prior[:,:,3:])
        self.assertEqual(stats['rank'],3)

    def test_residual_fit_reconstructs_identifiable_examples(self):
        ref=np.zeros((478,2));prior=np.zeros((478,2,52))
        q=np.eye(52)[:4];target=np.zeros((4,478,2))
        for i in range(4):target[i,:,1]=i*.01
        mapping,_=fit_mapping(ref,target,q,np.zeros(52),prior,1e-9)
        np.testing.assert_allclose(evaluate(ref,mapping,q,np.zeros(52)),target,atol=1e-9)

    def test_contact_has_correct_sign_and_preserves_input_and_outer_thickness(self):
        p=np.zeros((478,2));p[DOWN,1]=.1
        target=np.zeros_like(p)
        original=p.copy()
        corrected,amount=contact_correct(p,target,1)
        np.testing.assert_array_equal(p,original)
        np.testing.assert_allclose(gaps(corrected),.001,atol=1e-12)
        self.assertAlmostEqual(amount,.0495)
        unchanged,_=contact_correct(p,target,0)
        np.testing.assert_array_equal(unchanged,p)

    def test_local_coordinates_roundtrip_and_stable_alignment(self):
        rng=np.random.default_rng(9);ref=rng.normal(size=(478,2))
        ref[61]=[.4,.6];ref[291]=[.6,.62]
        frame=local_frame(ref)
        np.testing.assert_allclose(denormalize(normalize(ref,frame),frame),ref,atol=1e-12)
        theta=.13;r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        source=ref@r*1.4+[.1,.2]
        np.testing.assert_allclose(align_to_reference(source,ref),ref,atol=1e-12)

    def test_window_coverage_including_subwindow_and_tail(self):
        class Session:
            def run(self,names,inputs):
                self_test.assertEqual(inputs['pcm'].shape,(1,16000))
                return [np.zeros((1,60,52),np.float32)]
        self_test=self
        for samples in [1,120,3000,4000,8000,16000,23001]:
            with self.subTest(samples=samples):
                frames=infer_windows(Session(),np.zeros(samples,np.float32))
                self.assertEqual(frames.shape,((samples*60+15999)//16000,52))


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from local_mapping import fit_local,evaluate_local,adapt_lip_prior
from mapping import UP,DOWN
from source_registration import OUTER_UP,OUTER_DOWN,source_contact_mask,LIP_PAIRS


class LocalMappingTests(unittest.TestCase):
    def test_reference_is_exact_and_far_residual_returns_to_prior(self):
        ref=np.zeros((478,2));source_ref=np.array([.1,.2]);prior=np.zeros((478,2,2))
        coefficients=source_ref+np.array([[.3,0],[0,.4]])
        target=np.zeros((2,478,2));target[0,13,1]=.1;target[1,14,1]=.2
        m=fit_local(ref,target,coefficients,source_ref,prior,np.eye(2))
        np.testing.assert_allclose(evaluate_local(m,source_ref),ref,atol=1e-14)
        np.testing.assert_allclose(evaluate_local(m,np.array([100.,100.])),ref,atol=1e-14)
        self.assertLess(np.abs(evaluate_local(m,coefficients)-target).max(),.001)

    def test_anatomical_prior_scales_thickness_without_scaling_translation(self):
        source=np.zeros((478,2));target=np.zeros((478,2))
        outer=np.array(OUTER_UP[1:-1]);source[UP,1]=.05;target[UP,1]=.02
        prior=np.zeros((478,2,1));prior[UP,1,0]=.03;prior[outer,1,0]=-.01
        adjusted,_=adapt_lip_prior(target,source,prior)
        np.testing.assert_allclose((adjusted[UP,1,0]+adjusted[outer,1,0])/2,.01)
        np.testing.assert_allclose(adjusted[UP,1,0]-adjusted[outer,1,0],.016)

    def test_contact_labels_do_not_depend_on_projected_registration(self):
        count=int(LIP_PAIRS.max())+1;n=np.zeros((count,3));b=np.zeros((count,3,1))
        n[LIP_PAIRS[:,0],1]=.05;b[LIP_PAIRS[:,0],1,0]=.2
        reg=dict(source_neutral=n,source_basis=b,neutral=np.ones((478,3))*999)
        np.testing.assert_array_equal(source_contact_mask(reg,np.array([[0.],[1.]])),[True,False])


if __name__=='__main__':unittest.main()

"""Local Gaussian residual with a hard reference anchor and anatomical prior."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from mapping import UP,DOWN
from source_registration import OUTER_UP,OUTER_DOWN


def adapt_lip_prior(reference,source_reference_geometry,prior):
    """Transfer lip translation independently of fit-specific rim thickness.

    If the source changes its lip thickness by a fraction, apply that fraction
    to this face's thickness, while retaining mouth-width-scaled translation.
    """
    out=prior.copy();ratios=[]
    for inner,outer in [(UP,np.array(OUTER_UP[1:-1])),(DOWN,np.array(OUTER_DOWN[1:-1]))]:
        target=np.abs(reference[inner,1]-reference[outer,1])
        source=np.abs(source_reference_geometry[inner,1]-source_reference_geometry[outer,1])
        ratio=np.clip(target/np.maximum(source,.005),.1,5)
        translation=(prior[inner,1,:]+prior[outer,1,:])/2
        relative=(prior[inner,1,:]-prior[outer,1,:])*ratio[:,None]/2
        out[inner,1,:]=translation+relative;out[outer,1,:]=translation-relative
        ratios.extend(ratio.tolist())
    return out,dict(minimum_thickness_ratio=min(ratios),maximum_thickness_ratio=max(ratios))


def kernel(x,z,bandwidth):
    return np.exp(-cdist(x,z,'sqeuclidean')/(2*bandwidth*bandwidth))


def anchored_kernel(x,z,bandwidth):
    zero=np.zeros((1,x.shape[1]))
    return kernel(x,z,bandwidth)-kernel(x,zero,bandwidth)@kernel(z,zero,bandwidth).T


def fit_local(reference,target,coefficients,source_reference,prior,feature_basis,
              bandwidth_factor=1.,ridge=.001,confidence=None,ownership_mask=None):
    q=coefficients-source_reference;features=q@feature_basis.T
    distances=cdist(features,features)
    distances[distances<1e-6]=np.inf
    typical=np.median(distances.min(axis=1))
    bandwidth=max(.025,float(typical)*bandwidth_factor)
    matrix=anchored_kernel(features,features,bandwidth)
    residual=target-reference-np.einsum('nck,vk->vnc',prior,q)
    if ownership_mask is not None:residual*=ownership_mask[None,:,None]
    weights=np.ones(target.shape[:2]) if confidence is None else confidence
    solution=np.empty_like(residual);cache={}
    for vertex in range(len(reference)):
        w=np.maximum(weights[:,vertex],.02);key=w.tobytes()
        if key not in cache:
            cache[key]=cho_factor(matrix+np.diag(ridge/w))
        solution[:,vertex]=cho_solve(cache[key],residual[:,vertex])
    return dict(reference=reference,source_reference=source_reference,prior=prior,
                feature_basis=feature_basis,train_features=features,
                bandwidth=np.array(bandwidth),solution=solution,
                ridge=np.array(ridge),bandwidth_factor=np.array(bandwidth_factor))


def evaluate_local(model,coefficients):
    c=np.asarray(coefficients);single=c.ndim==1;c=np.atleast_2d(c)
    q=c-model['source_reference'];features=q@model['feature_basis'].T
    k=anchored_kernel(features,model['train_features'],float(model['bandwidth']))
    out=model['reference']+np.einsum('nck,tk->tnc',model['prior'],q)+np.einsum('tv,vnc->tnc',k,model['solution'])
    return out[0] if single else out

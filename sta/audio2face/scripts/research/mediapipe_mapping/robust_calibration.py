"""Treat fitted visemes as noisy measurements, preserving original assets.

Only isolated contour anomalies and explicitly closed-pose aperture are repaired.
Coherent face-specific expression differences are not classified as corruption
merely because they differ from the shared prior.
"""
import numpy as np
from mapping import (ANCHORS,LIPS,UP,DOWN,align_to_reference,fit_mapping,
                     INNER_UP,INNER_DOWN,OUTER_UP,OUTER_DOWN)

CLOSED_POSES={'PP','lip_press','teeth_mouth_closed'}


def robust_alignment(points,reference):
    x=points[ANCHORS,:2];y=reference[ANCHORS,:2]
    width=np.linalg.norm(reference[291,:2]-reference[61,:2])
    weights=np.ones(len(ANCHORS));result=points[:,:2].copy()
    for _ in range(10):
        xc=np.average(x,axis=0,weights=weights);yc=np.average(y,axis=0,weights=weights)
        x0=x-xc;y0=y-yc
        u,_,vt=np.linalg.svd((x0*weights[:,None]).T@y0)
        rotation=u@vt
        if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@vt
        scale=np.sum(weights[:,None]*(x0@rotation)*y0)/np.sum(weights[:,None]*x0*x0)
        translation=yc-xc@rotation*scale
        result=points[:,:2]@rotation*scale+translation
        error=np.linalg.norm(result[ANCHORS]-y,axis=1)
        weights=np.minimum(1,(.025*width)/np.maximum(error,1e-12))**2
    return result


def prepare_observations(poses):
    reference=poses['neutral'];prepared={k:v.copy() for k,v in poses.items()}
    confidence={k:np.ones(len(reference)) for k in poses};notes={}
    for name,raw in poses.items():
        if name=='neutral':continue
        p=prepared[name];repaired=set()
        # The four curves are anatomical correspondences. Detect *isolated*
        # changes in local curvature, not entire distinctive mouth shapes.
        for contour in [INNER_UP,INNER_DOWN,OUTER_UP,OUTER_DOWN]:
            c=np.array(contour);r=reference[c];values=raw[c]
            steps=np.linalg.norm(np.diff(r,axis=0),axis=1)
            t=np.r_[0,np.cumsum(np.maximum(steps,1e-6))]
            anomalies=np.zeros(len(c));expected={}
            for j in range(1,len(c)-1):
                fraction=(t[j]-t[j-1])/(t[j+1]-t[j-1])
                reference_interp=r[j-1]*(1-fraction)+r[j+1]*fraction
                prediction=values[j-1]*(1-fraction)+values[j+1]*fraction+(r[j]-reference_interp)
                anomalies[j]=np.linalg.norm(values[j]-prediction);expected[j]=prediction
            for j in range(1,len(c)-1):
                if anomalies[j]>.08 and anomalies[j]>1.25*max(anomalies[j-1],anomalies[j+1]):
                    index=c[j];p[index]=expected[j];confidence[name][index]=.2;repaired.add(int(index))
        spatial_change=float(np.max(np.linalg.norm(p-raw,axis=1)))
        closure_change=0.
        if name in CLOSED_POSES:
            gap=p[DOWN,1]-p[UP,1]
            # One millimetre is not a universal avatar displacement. This is a
            # normalized small residual aperture, below the existing 1% limit.
            shift=np.clip((.003-gap)/2,-.05,.05)
            p[UP,1]-=shift;p[DOWN,1]+=shift
            p[np.array(OUTER_UP[1:-1]),1]-=shift
            p[np.array(OUTER_DOWN[1:-1]),1]+=shift
            closure_change=float(np.max(np.abs(shift)))
        notes[name]=dict(isolated_repaired_vertices=sorted(repaired),
            maximum_spatial_measurement_repair=spatial_change,
            maximum_closed_pose_repair=closure_change,
            observation_lip_rms=float(np.sqrt(np.mean(np.sum((p[LIPS]-raw[LIPS])**2,axis=1)))))
    return prepared,confidence,notes


def fit_robust_mapping(reference,target,coefficients,source_reference,prior,
                       ridge=.03,confidence=None,iterations=4):
    """Point-wise Huber IRLS with expression-adaptive scale and regularized prior.

    Wide but spatially coherent expressions raise their own robust scale instead
    of being discarded as outliers. Calibration repairs are separately recorded.
    """
    q=coefficients-source_reference
    residual=target-reference-np.einsum('nck,vk->vnc',prior,q)
    base=np.ones(target.shape[:2]) if confidence is None else confidence.copy()
    weights=base.copy();mapping=prior.copy();stats={}
    for _ in range(iterations):
        cache={}
        correction=np.empty_like(prior)
        for vertex in range(len(reference)):
            w=weights[:,vertex];key=np.round(w,8).tobytes()
            if key not in cache:
                root=np.sqrt(w);x=q*root[:,None]
                u,s,vt=np.linalg.svd(x,full_matrices=False)
                cache[key]=(vt.T*(s/(s*s+ridge)))@u.T*root[None,:]
            correction[vertex]=residual[:,vertex,:].T@cache[key].T
        mapping=prior+correction
        predicted=reference+np.einsum('nck,vk->vnc',mapping,q)
        error=np.linalg.norm(predicted-target,axis=-1)
        scale=np.maximum(.03,2.5*np.median(error[:,LIPS],axis=1))
        weights=base*np.minimum(1,scale[:,None]/np.maximum(error,1e-12))
    _,s,_=np.linalg.svd(q,full_matrices=False);supported=s>1e-3
    stats=dict(rank=int(supported.sum()),singular_values=s.tolist(),
        condition=float(s[0]/s[supported][-1]) if supported.any() else None,ridge=ridge,
        robust_iterations=iterations,downweighted_lip_observations=int(np.sum(weights[:,LIPS]<.5)),
        minimum_lip_weight=float(weights[:,LIPS].min()))
    return mapping,stats

"""Geometry-conditioned source-pose correspondence, developed without test faces.

The shared source poses are bounded rig fits, not claimed to be raw network
outputs. Final speech checks still use actual, held-out network output.
"""
import numpy as np
from scipy.optimize import lsq_linear
from mapping import LIPS,UP,DOWN,local_frame,normalize

DEVELOPMENT_FACES=['01d55f04','dd80126a','0c72c586','3bbd86ff']
NEW_TEST_FACES=['5a02aa84','c4e71fe5','e76f3815','ff5687da']
FRESH_TEST_FACES=['201c8934','679803f3','8f228898','dac8814d']
ADDITIONAL_POSES=['TH','DD','kk','CH','SS','nn','RR','relaxed_mouth_open',
                  'lip_press','small_smile','big_smile','smirk_left','smirk_right',
                  'teeth_mouth_closed','teeth_mouth_open']


def geometry_features(points):
    """Linear detailed lip displacement, aperture profile, spread and lip center.

    Coordinates are in each face's own neutral mouth frame and mouth widths.
    Depth is deliberately excluded for fitted 2.5D target images.
    """
    p=np.asarray(points)
    coords=p[...,LIPS,:2].reshape(*p.shape[:-2],-1)*.35
    aperture=(p[...,DOWN,1]-p[...,UP,1])*2
    spread=(p[...,291,0]-p[...,61,0])[...,None]
    centers=np.stack([p[...,UP,1].mean(-1),p[...,DOWN,1].mean(-1)],axis=-1)
    return np.concatenate([coords,aperture,spread,centers],axis=-1)


def fit_source_pose(prior,basis,reference_weights,desired_displacement,owned,
                    coefficient_regularization=.001):
    """Bounded source coefficients, fixed geometry objective and prior strength."""
    # Exact linear Jacobian: evaluate each coefficient's geometry displacement.
    feature_basis=np.stack([geometry_features(basis[:,:,i]) for i in range(52)],axis=1)
    active=np.flatnonzero(owned)
    a=feature_basis[:,active]
    desired=geometry_features(desired_displacement)
    # Unknown is absolute w, keeping the zero/neutral centering explicit.
    b=desired+a@reference_weights[active]
    regularizer=np.sqrt(coefficient_regularization)
    result=lsq_linear(np.vstack([a,np.eye(len(active))*regularizer]),
                      np.r_[b,prior[active]*regularizer],bounds=(0,1),
                      tol=1e-10,max_iter=200)
    if not result.success:raise ValueError('Bounded source geometry matching failed: '+result.message)
    weights=reference_weights.copy();weights[active]=result.x
    residual=a@(weights[active]-reference_weights[active])-desired
    return weights,dict(geometry_rms=float(np.sqrt(np.mean(residual**2))),
                        optimality=float(result.optimality),iterations=result.nit,
                        coefficients_at_boundary=int(np.sum((result.x<1e-5)|(result.x>1-1e-5))))


def build_geometry_library(legacy_library,source_n,source_d,source_reference,
                           avatars,face_loader,roles,channels):
    reference_pose=source_n+np.einsum('nck,k->nc',source_d,source_reference)
    frame=local_frame(reference_pose)
    basis=np.einsum('nck,cd->ndk',source_d[:,:2],frame[2])/frame[1]
    owned=np.array([str(n).startswith(('mouth','jaw')) or str(n)=='cheekPuff' for n in channels])
    examples={role:[] for role in roles}
    for face in DEVELOPMENT_FACES:
        _,_,poses,_,_=face_loader(avatars/face/'face-meshes.json')
        for role in roles:examples[role].append(poses[role]-poses['neutral'])
    library={}
    observed=np.array([v['coefficients'] for v in legacy_library.values()])
    observed_geometry=np.einsum('nck,vk->vnc',basis,observed-source_reference)
    seed_features=geometry_features(observed_geometry)
    for role in roles:
        desired=np.median(np.array(examples[role]),axis=0)
        # Geometry alone chooses the observed seed, including for nonphonemic
        # expression examples. Labels identify target files, not source vectors.
        seed_index=int(np.argmin(np.mean((seed_features-geometry_features(desired))**2,axis=1)))
        seed_role=list(legacy_library)[seed_index]
        seed=observed[seed_index]
        w,diagnostics=fit_source_pose(seed,basis,source_reference,desired,owned)
        library[role]=dict(coefficients=w.tolist(),kind='geometry-matched bounded source-rig pose',
                           development_faces=DEVELOPMENT_FACES,
                           feature_match=diagnostics,
                           desired_displacement=desired.tolist(),
                           seed_stem=legacy_library[seed_role]['stem'],
                           seed_frame=legacy_library[seed_role]['frame'],
                           seed_selection='nearest observed seed by geometric feature distance')
    return library


def choose_contact_reference(poses):
    """Pick a valid same-fit closed pose; never manufacture a new reference."""
    from mapping import gaps
    candidates=[]
    for name in ['PP','lip_press','teeth_mouth_closed','sil','neutral']:
        if name not in poses:continue
        p=poses[name];g=gaps(p)
        upper_thickness=p[UP,1]-p[[185,40,39,37,0,267,269,270,409],1]
        lower_thickness=p[[146,91,181,84,17,314,405,321,375],1]-p[DOWN,1]
        width=np.linalg.norm(p[291]-p[61])
        valid=bool(g.max()<=.01 and g.min()>=-.005 and .65<=width<=1.3
                   and upper_thickness.min()>0 and lower_thickness.min()>0)
        candidates.append(dict(name=name,valid=valid,max_gap=float(g.max()),
                               min_gap=float(g.min()),min_thickness=float(min(upper_thickness.min(),lower_thickness.min()))))
    valid=[x for x in candidates if x['valid']]
    if valid:
        selected=min(valid,key=lambda x:max(abs(x['max_gap']),abs(x['min_gap'])))
        return selected['name'],True,candidates
    return 'PP',False,candidates


def source_feature_basis(source_n,source_d,source_reference,owned):
    """Measure actual source 3D lip movement, not coefficient-vector distance."""
    ref=source_n+np.einsum('nck,k->nc',source_d,source_reference)
    frame=local_frame(ref)
    local=source_d.copy()/frame[1]
    local[:,:2]=np.einsum('nck,cd->ndk',source_d[:,:2],frame[2])/frame[1]
    # Source depth is known rig geometry. It is an input feature, never asserted
    # to be reliable depth in the avatar's 2.5D landmark fits.
    features=[]
    for i in range(52):
        features.append(np.r_[geometry_features(local[:,:,i]),local[LIPS,2,i]*.35])
    return np.array(features).T*owned[None,:]


def fit_geometric_residual(reference,target,coefficients,source_reference,prior,
                           feature_basis,ridge=.03):
    """Compile a residual based on source deformation back to a 52-control map.

    Source coefficient combinations with equal physical features receive equal
    residuals. Unsupported feature directions retain the source prior.
    """
    from mapping import fit_mapping
    q=coefficients-source_reference
    x=q@feature_basis.T
    residual=target-reference-np.einsum('nck,vk->vnc',prior,q)
    feature_map,stats=fit_mapping(np.zeros_like(reference),residual,x,
        np.zeros(feature_basis.shape[0]),np.zeros((*reference.shape,feature_basis.shape[0])),ridge)
    correction=np.einsum('ncf,fk->nck',feature_map,feature_basis)
    stats['metric']='source lip deformation features (image plane and known source depth)'
    return prior+correction,stats

"""Fit-specific image-plane transfer of source cheek deformation (no new expressions)."""
import numpy as np

LIPS=[78,191,80,81,82,13,312,311,310,415,308,95,88,178,87,14,317,402,318,324,
      61,185,40,39,37,0,267,269,270,409,291,146,91,181,84,17,314,405,321,375]

def smoothstep(a,b,x):
    t=np.clip((x-a)/(b-a),0,1)
    return t*t*(3-2*t)

def build_cheek_transfer(reg,model,edges):
    n=reg['neutral'][:,:2].copy();d=reg['basis'][:,:2,:].copy()
    n[:,1]*=-1;d[:,1,:]*=-1
    pose=n+np.einsum('nck,k->nc',d,model['source_reference'])
    axis=pose[291]-pose[61];width=np.linalg.norm(axis);x=axis/width
    rotation=np.array([x,[-x[1],x[0]]]).T
    source=(pose-(pose[61]+pose[291])/2)@rotation/width
    delta=np.einsum('nck,cd->ndk',d,rotation)/width
    owned=np.array([str(c).startswith(('mouth','jaw')) or str(c)=='cheekPuff' for c in reg['channels']])
    delta*=owned[None,None,:]
    target=model['reference'];adj=[set() for _ in range(478)]
    for a,b in edges:
        if a<468 and b<468:adj[a].add(b);adj[b].add(a)
    transferred=np.zeros_like(delta);conditions=[]
    for i in range(468):
        neighbors=set(adj[i])
        for j in list(neighbors):neighbors.update(adj[j])
        neighbors.discard(i);ids=np.array(sorted(neighbors),int)
        if len(ids)<3:continue
        a=source[ids]-source[i];b=target[ids]-target[i]
        base=1/np.maximum(np.linalg.norm(a,axis=1),.03)**2
        weights=base.copy();jac=np.eye(2)
        # Robust local affine fit, including translation. Centering prevents
        # a single bad center landmark from biasing every neighbor displacement.
        for _ in range(4):
            ac=np.average(a,axis=0,weights=weights);bc=np.average(b,axis=0,weights=weights)
            aa=a-ac;bb=b-bc
            gram=(aa.T*weights)@aa;ridge=max(.01*np.trace(gram),1e-8)
            jac=np.linalg.solve(gram+ridge*np.eye(2),(aa.T*weights)@bb+ridge*np.eye(2))
            error=np.linalg.norm(aa@jac-bb,axis=1)
            scale=max(.015,2.5*np.median(error))
            weights=base*np.minimum(1,scale/np.maximum(error,1e-12))
        u,s,vt=np.linalg.svd(jac);jac=(u*np.clip(s,.35,2.5))@vt
        if np.linalg.det(jac)<=0:jac=np.eye(2)
        transferred[i]=jac.T@delta[i];conditions.append(float(np.linalg.cond(gram+ridge*np.eye(2))))
    # Compact smooth patches, positioned by this avatar's own landmarks.
    blend=np.zeros(478)
    for cheek in [205,425]:
        cx,cy=target[cheek];eye=145 if cheek==205 else 374
        # Fit-specific radius and eye boundary; no fixed image-space neutral.
        side=61 if cheek==205 else 291
        rx=max(abs(cx-target[side,0])*2.5,.65)
        ry=max(abs(target[152,1]-target[eye,1])*.55,.65)
        radius=np.sqrt(((target[:,0]-cx)/rx)**2+((target[:,1]-cy)/ry)**2)
        patch=(1-smoothstep(0,1,radius))*smoothstep(target[eye,1]+.04,target[eye,1]+.3,target[:,1])
        patch*=smoothstep(.35,.65,np.abs(target[:,0]))
        blend=np.maximum(blend,patch)
    blend[LIPS]=0;blend[468:]=0
    blend[target[:,1]<=min(target[145,1],target[374,1])+.04]=0
    return dict(basis=transferred.astype(np.float32).ravel().tolist(),blend=blend.astype(np.float32).tolist(),
        method='bounded local affine source-rig deformation transfer; fit-specific cheek patches',
        maximumLocalCondition=max(conditions))

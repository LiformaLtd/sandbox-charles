"""Additional conservative image-plane checks; never mutates avatar assets."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from mapping import LIPS,UP,DOWN,evaluate,contact_correct
from source_registration import OUTER_UP,OUTER_DOWN,source_contact_mask
from run_probe import face_data,MODEL_SHA,write_json,sha


def triangles_from_edges(edges):
    """Mirror the Player graph-clique construction and non-manifold filtering."""
    adj=defaultdict(list)
    for a,b in edges:
        if a==b:continue
        adj[a].append(b);adj[b].append(a)
    seen=set();triangles=[]
    for a,b in edges:
        if a==b:continue
        u,v=sorted([a,b]);nu,nv=adj[u],adj[v]
        small,large=(nu,set(nv)) if len(nu)<=len(nv) else (nv,set(nu))
        for w in small:
            if w in [u,v] or w not in large:continue
            t=tuple(sorted([u,v,w]))
            if t not in seen:seen.add(t);triangles.append(t)
    def links(t):return [tuple(sorted([t[a],t[b]])) for a,b in [(0,1),(1,2),(2,0)]]
    removed=set()
    while True:
        count=Counter(e for i,t in enumerate(triangles) if i not in removed for e in links(t))
        worst=None;score=0
        for i,t in enumerate(triangles):
            if i in removed:continue
            current=sum(max(count[e]-2,0) for e in links(t))
            if current>score:worst,score=i,current
        if worst is None:break
        removed.add(worst)
    return np.array([t for i,t in enumerate(triangles) if i not in removed and max(t)<478],int)


def signed_areas(points,triangles):
    a=points[...,triangles[:,1],:]-points[...,triangles[:,0],:]
    b=points[...,triangles[:,2],:]-points[...,triangles[:,0],:]
    return (a[...,0]*b[...,1]-a[...,1]*b[...,0])/2


def lip_thickness(points):
    top=points[...,UP,1]-points[...,np.array(OUTER_UP[1:-1]),1]
    bottom=points[...,np.array(OUTER_DOWN[1:-1]),1]-points[...,DOWN,1]
    return np.concatenate([top,bottom],axis=-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--avatars',type=Path,required=True);p.add_argument('--sandbox-assets',type=Path,required=True)
    args=p.parse_args();r=args.run;d=json.loads((r/'results.json').read_text())
    study=args.sandbox_assets/'compact-v4-confirmation'
    rows=[row for row in json.loads((study/'manifest.json').read_text()) if row['voice'] in ['M3','F3']]
    tracks=[];track_hashes={}
    for row in rows:
        path=study/'predictions'/MODEL_SHA/(row['sha256_audio']+'.npy')
        tracks.append(np.load(path));track_hashes[str(path)]=sha(path)
    w=np.concatenate(tracks);ref=np.array(json.loads((r/'source-library.json').read_text())['reference'])
    reg=np.load(r/'source-registration.npz');source_n=reg['neutral'].copy();source_d=reg['basis'].copy()
    source_n[:,1]*=-1;source_d[:,1]*=-1
    source=source_n+np.einsum('nck,tk->tnc',source_d,w)
    seal=source_contact_mask(reg,w)
    owned=np.array([str(n).startswith(('mouth','jaw')) or str(n)=='cheekPuff' for n in reg['channels']])
    distance=np.linalg.norm((w-ref)*owned,axis=1)
    protocol=dict(area_epsilon=1e-7,thickness_epsilon=.005,
                  scope='lip-adjacent triangles only, same topology construction as Player; image plane before production compositor',
                  rule='No new inverted nondegenerate lip-adjacent triangle relative to the fit reference')
    write_json(r/'safety-protocol.json',protocol)
    results=[]
    for f in d['faces']:
        path=args.avatars/f['id']/'face-meshes.json'
        data,_,poses,*_=face_data(path)
        triangles=triangles_from_edges(data['faceMeshEdges'])
        triangles=triangles[np.any(np.isin(triangles,LIPS),axis=1)]
        model=np.load(r/(f['id']+'-mapping.npz'))
        raw=evaluate(model['reference'],model['basis'],w,ref)
        contact=f.get('contact_reference','PP');corrected=[]
        for points,is_seal,dist in zip(raw,seal,distance):
            corrected.append(contact_correct(points,poses[contact],float(is_seal)*min(1,float(dist)/.15))[0])
        corrected=np.array(corrected)
        base=signed_areas(model['reference'],triangles)
        valid=np.abs(base)>protocol['area_epsilon']
        def check(points):
            area=signed_areas(points,triangles)
            flips=(area[...,valid]*np.sign(base[valid])) < -protocol['area_epsilon']
            thin=lip_thickness(points)
            return dict(frame_fraction_with_flips=float(np.mean(np.any(flips,axis=-1))),
                        max_flipped_triangles=int(np.max(flips.sum(-1))),
                        frame_fraction_with_negative_lip_thickness=float(np.mean(np.min(thin,axis=-1)<-.005)),
                        minimum_lip_thickness=float(thin.min()),finite=bool(np.isfinite(points).all()))
        raw_result=check(raw);fixed_result=check(corrected)
        source_examples=np.array([poses[k] for k in f['calibration_roles']])
        example_result=check(source_examples)
        result=dict(id=f['id'],lip_adjacent_triangles=len(triangles),nondegenerate_reference_triangles=int(valid.sum()),
                    source_examples=example_result,raw=raw_result,corrected=fixed_result,
                    conservative_geometry_pass=fixed_result['frame_fraction_with_flips']==0
                        and fixed_result['frame_fraction_with_negative_lip_thickness']==0 and fixed_result['finite'])
        results.append(result)
        print(f['id'],'raw flip frames',round(raw_result['frame_fraction_with_flips']*100,1),
              'corrected',round(fixed_result['frame_fraction_with_flips']*100,1),
              'negative thickness frames',round(fixed_result['frame_fraction_with_negative_lip_thickness']*100,1),flush=True)
    write_json(r/'geometry-safety.json',dict(protocol=protocol,prediction_hashes=track_hashes,faces=results,
                     all_pass=all(f['conservative_geometry_pass'] for f in results),
                     note='A pass would not cover depth, smoothing, texture warping or the production oral/blink compositor.'))


if __name__=='__main__':main()

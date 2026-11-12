"""Known corruption tests on in-memory copies of actual avatar observations."""
import argparse,json
from pathlib import Path
import numpy as np
from mapping import (LIPS,UP,DOWN,ANCHORS,local_frame,normalize,evaluate,fit_mapping,
                     align_to_reference)
from robust_calibration import (prepare_observations,fit_robust_mapping,robust_alignment)
from run_probe import face_data,source_xy,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--avatars',type=Path,required=True);p.add_argument('--sandbox-assets',type=Path,required=True)
    args=p.parse_args();r=args.run;results=json.loads((r/'results.json').read_text())
    library=json.loads((r/'source-library.json').read_text());reg=np.load(r/'source-registration.npz')
    c=np.array(library['reference']);n,b=source_xy(reg);source=n+np.einsum('nck,k->nc',b,c)
    sf=local_frame(source);prior=np.einsum('nck,cd->ndk',b[:,:2],sf[2])/sf[1]
    sp=normalize(source,sf);mask=np.clip((sp[:,1]+.45)/.25,0,1)*np.clip((1.5-np.abs(sp[:,0]))/.4,0,1);mask[468:]=0
    owned=np.array([str(x).startswith(('mouth','jaw')) or str(x)=='cheekPuff' for x in reg['channels']])
    prior*=mask[:,None,None]*owned[None,None,:]
    study=args.sandbox_assets/'compact-v4-confirmation';rows=[v for v in json.loads((study/'manifest.json').read_text()) if v['voice'] in ['M3','F3']]
    tracks=np.concatenate([np.load(study/'predictions'/results['protocol']['model_sha256']/(v['sha256_audio']+'.npy')) for v in rows])[::10]
    protocol=dict(seed=73,lip_point_spike=.25,false_closed_pose_added_gap=.08,
                  alignment_anchor_offset=[.25,-.2],sampled_frame_stride=10,
                  maximum_stability_lip_rms=.03,
                  comparison='Corrupted-input mapping versus the same method on uncorrupted observations; this measures robustness, not absolute animation quality')
    write_json(r/'noise-protocol.json',protocol)
    output=[]
    def build(poses,keys,coeff,robust):
        if robust:
            poses,confidence,_=prepare_observations(poses)
            m,_=fit_robust_mapping(poses['neutral'],np.array([poses[k] for k in keys]),coeff,c,prior,
                                  confidence=np.array([confidence[k] for k in keys]))
        else:m,_=fit_mapping(poses['neutral'],np.array([poses[k] for k in keys]),coeff,c,prior)
        return m*mask[:,None,None]
    for face in results['faces']:
        _,_,raw,_,_=face_data(args.avatars/face['id']/'face-meshes.json',True)
        keys=face['calibration_roles'];coeff=np.array([library['roles'][k]['coefficients'] for k in keys]);coeff=np.where(owned,coeff,c)
        baselines={mode:build(raw,keys,coeff,mode=='robust') for mode in ['ordinary','robust']}
        tests=[]
        for case in ['lip_vertex_spike','false_closed_gap','gaussian_lip_noise']:
            corrupted={k:v.copy() for k,v in raw.items()}
            if case=='lip_vertex_spike':corrupted['E'][13,1]+=.25
            elif case=='false_closed_gap':
                corrupted['PP'][UP,1]-=.04;corrupted['PP'][DOWN,1]+=.04
                corrupted['PP'][np.array([185,40,39,37,0,267,269,270,409]),1]-=.04
                corrupted['PP'][np.array([146,91,181,84,17,314,405,321,375]),1]+=.04
            else:
                rng=np.random.default_rng(73)
                for k in keys:corrupted[k][LIPS]+=rng.normal(0,.01,size=(len(LIPS),2))
            metrics={}
            for mode in baselines:
                m=build(corrupted,keys,coeff,mode=='robust')
                difference=np.einsum('nck,tk->tnc',(m-baselines[mode])[LIPS],tracks-c)
                metrics[mode]=dict(lip_rms=float(np.sqrt(np.mean(np.sum(difference**2,axis=-1)))),
                    maximum_point_change=float(np.max(np.linalg.norm(difference,axis=-1))))
            tests.append(dict(case=case,methods=metrics))
        # Known in-plane transform plus one bad eye/nose anchor: correct output
        # is the original same-face reference, independently known here.
        ref=raw['neutral'];transformed=ref*1.1+[.2,.1]
        transformed[ANCHORS[0]]+=[.25,-.2]
        ordinary=align_to_reference(transformed,ref);robust=robust_alignment(transformed,ref)
        alignment=dict(ordinary_lip_rms=float(np.sqrt(np.mean(np.sum((ordinary[LIPS]-ref[LIPS])**2,axis=-1)))),
                       robust_lip_rms=float(np.sqrt(np.mean(np.sum((robust[LIPS]-ref[LIPS])**2,axis=-1)))))
        passed=all(t['methods']['robust']['lip_rms']<=.03 for t in tests) and alignment['robust_lip_rms']<=.03
        row=dict(id=face['id'],corruption_tests=tests,anchor_corruption=alignment,stability_pass=passed)
        output.append(row)
        print(face['id'],'noise stability',passed,'spike RMS ordinary/robust',
              round(tests[0]['methods']['ordinary']['lip_rms'],4),round(tests[0]['methods']['robust']['lip_rms'],4),flush=True)
    write_json(r/'noise-results.json',dict(protocol=protocol,faces=output,all_pass=all(v['stability_pass'] for v in output),
        limitations='No guarantee for a coherent but entirely incorrect mesh or a corrupted identity reference; those require image evidence or additional trusted observations.'))


if __name__=='__main__':main()

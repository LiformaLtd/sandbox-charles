"""Reproducible mapping feasibility experiment on actual, read-only avatar assets."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from source_registration import build_registration, load_rig, render_source,source_contact_mask
from mapping import (LIPS, UP, DOWN, ANCHORS, INNER_UP, INNER_DOWN, OUTER_UP, OUTER_DOWN,
                     align_to_reference, local_frame, normalize, denormalize,
                     gaps, mouth_metrics, fit_mapping, evaluate, contact_correct, lip_error)

MODEL_SHA = 'b4a275acf78b982fb6ec6fc55333c5cf25613e72b6d9b3d80e95d6e74ac6261d'
ROLES = {'PP': ['m','b','p'], 'aa': ['aa','ae','ah'], 'E': ['eh','ey'],
         'ih': ['ih','iy'], 'oh': ['ow','ao'], 'ou': ['uw','w'], 'FF': ['f','v']}
FIT_IDS = ['01d55f04','dd80126a','0c72c586','3bbd86ff',
           '394d2cb9','c1bb51a7','d3194283','64cf2cc1',
           '05a87620','05a87620_old','05a87620_old2']
PROTOCOL = dict(version=3, model_sha256=MODEL_SHA, face_ids=FIT_IDS,
                source_example_voices=['Trevor','Pippa'], held_out_voices=['M3','F3'],
                roles=ROLES, ridge=.03, calibration_lip_rms_limit=.03,
                held_out_viseme_lip_rms_limit=.06, sealed_gap_limit=.01,
                penetration_limit=-.005, source_contact_threshold_cm=.1,
                max_contact_correction=.05, rest_protection_coefficient_distance=.15,
                owned_channels='jaw*, mouth*, cheekPuff',
                contact_eligibility='source seal and owned-coefficient distance from reference >= 0.15; rest is a separate invariant',
                sensitivity_ridges=[.0001,.001,.003,.01,.03,.1],
                description='Image-plane feasibility screen, not full player/compositor validation; limits in neutral mouth widths')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def infer_windows(session, pcm):
    """Exact sandbox sample grid and central crop, including short/tail padding."""
    tracks = []
    for start in range(-16000, len(pcm), 8000):
        if start >= len(pcm)-4000:
            break
        samples = np.zeros(16000, np.float32)
        left, right = max(0,start), min(start+16000,len(pcm))
        if right > left:
            samples[left-start:right-start] = pcm[left:right]
        output = session.run(['blendshapes'], {'pcm':samples[None]})[0].reshape(60,52)
        for i in range(15,45):
            index=start+i*16000//60
            if 0 <= index < len(pcm):
                tracks.append(output[i])
    return np.array(tracks)


def source_xy(reg):
    n, d = reg['neutral'].copy(), reg['basis'].copy()
    n[:,1] *= -1; d[:,1] *= -1
    return n, d


def build_library(rows, study, reg, source_reference, out):
    candidates = defaultdict(list)
    source_n, source_d = source_xy(reg)
    sf = local_frame(source_n)
    source_n_norm = normalize(source_n,sf)
    source_d_norm = np.einsum('nck,cd->ndk',source_d[:,:2],sf[2])/sf[1]
    cache=study/'predictions'/MODEL_SHA
    hashes={}
    for row in rows:
        if row['voice'] not in PROTOCOL['source_example_voices']:
            continue
        path=cache/(row['sha256_audio']+'.npy')
        w=np.load(path); hashes[str(path)]=sha(path)
        geom=source_n_norm+np.einsum('nck,tk->tnc',source_d_norm,w)
        gap=np.median(gaps(geom),axis=1)
        for phone in row['phones']:
            roles=[r for r,phones in ROLES.items() if phone['phone'] in phones]
            if not roles: continue
            # Source contact can anticipate the acoustic event. Include the
            # established association interval, then select actual source poses.
            a=max(0,int((phone['start']-(.1 if roles[0]=='PP' else 0))*60))
            b=min(len(w),max(a+1,int(phone['end']*60)))
            if a>=len(w): continue
            for role in roles:
                ids=np.arange(a,b)
                if role=='PP': ids=ids[gap[ids]*sf[1] <= .1]
                if not len(ids): continue
                # One actual frame per event; do not average into an unobserved pose.
                idx=int(ids[np.argmin(gap[ids])] if role=='PP' else ids[len(ids)//2])
                candidates[role].append(dict(w=w[idx].astype(float), geom=geom[idx],
                    stem=row['stem'], frame=idx, voice=row['voice']))
    library={}
    for role in ROLES:
        pool=candidates[role]
        if not pool: raise ValueError(f'No observed source candidates for {role}')
        metrics=np.array([[np.median(gaps(x['geom'])),
                           np.linalg.norm(x['geom'][291]-x['geom'][61]),
                           np.mean(x['geom'][LIPS,1])] for x in pool])
        center=np.median(metrics,axis=0)
        if role=='aa': center[0]=np.quantile(metrics[:,0],.8)
        scale=np.maximum(np.std(metrics,axis=0),.01)
        picked=pool[np.argmin(np.sum(((metrics-center)/scale)**2,axis=1))]
        library[role]=dict(coefficients=picked['w'].tolist(), stem=picked['stem'],
                           frame=picked['frame'], voice=picked['voice'],
                           candidate_count=len(pool), source_metrics=mouth_metrics(picked['geom']))
    write_json(out/'source-library.json',dict(reference=source_reference.tolist(),
                reference_geometry=normalize(source_n+np.einsum('nck,k->nc',source_d,source_reference),sf).tolist(),
                roles=library, prediction_hashes=hashes,
                registration_note='Automatic source registration is preliminary and requires independent review.'))
    return library,source_n,source_d


def face_data(path,robust_alignment_enabled=False):
    data=json.loads(path.read_text())
    selected={v['id']:v for v in data['visemes'] if len(v.get('landmarks',[]))==478}
    neutral=np.array(selected['neutral'].get('cohortLandmarks',selected['neutral']['landmarks']))
    frame=local_frame(neutral)
    poses={}; fit_rms={}
    for key,v in selected.items():
        raw=np.array(v.get('cohortLandmarks',v['landmarks']))
        if robust_alignment_enabled:
            from robust_calibration import robust_alignment
            aligned=robust_alignment(raw,neutral)
        else:aligned=align_to_reference(raw,neutral)
        poses[key]=normalize(aligned,frame)
        fit_rms[key]=float(np.sqrt(np.mean((aligned[:,:2]-raw[:,:2])**2)))
    return data, selected, poses,frame,fit_rms


def annotated(image, raw, predicted, frame, label, expected):
    canvas=image.copy().convert('RGB'); draw=ImageDraw.Draw(canvas)
    # Cohort geometry belongs to the shared reference, whereas image pixels live
    # in each plate's original frame. Estimate only the stable-anchor transform.
    canonical=denormalize(expected,frame)
    a=np.column_stack([canonical[ANCHORS],np.ones(len(ANCHORS))])
    transform=np.linalg.lstsq(a,raw[ANCHORS,:2],rcond=None)[0]
    mapped=np.column_stack([denormalize(predicted,frame),np.ones(len(predicted))])@transform
    for points,color in [(raw,(30,230,140)),(mapped,(255,70,100))]:
        pts=points[:,:2]*np.array(canvas.size)
        for line in [OUTER_UP,OUTER_DOWN,INNER_UP,INNER_DOWN]:
            draw.line([tuple(p) for p in pts[line]],fill=color,width=3)
    lip=raw[LIPS,:2]*np.array(canvas.size)
    mid=lip.mean(0); radius=max(np.ptp(lip[:,0])*1.1,75)
    box=(int(mid[0]-radius),int(mid[1]-radius*.7),int(mid[0]+radius),int(mid[1]+radius*.7))
    crop=canvas.crop(box).resize((440,280))
    tile=Image.new('RGB',(440,320),'#151921');tile.paste(crop,(0,40))
    ImageDraw.Draw(tile).text((10,12),label,fill='white')
    return tile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--avatars',type=Path,required=True)
    parser.add_argument('--sandbox-assets',type=Path,required=True)
    parser.add_argument('--source-rig',type=Path,required=True)
    parser.add_argument('--face-task',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--assignment',choices=['legacy','geometry'],default='legacy')
    parser.add_argument('--calibration-set',choices=['core','extended'],default='core')
    parser.add_argument('--registration',choices=['legacy','curves'],default='legacy')
    parser.add_argument('--robust',action='store_true')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=dict(PROTOCOL);fit_ids=list(FIT_IDS);roles=dict(ROLES)
    if args.assignment=='geometry':
        from geometry_match import DEVELOPMENT_FACES,NEW_TEST_FACES
        fit_ids+=NEW_TEST_FACES
        if args.calibration_set=='extended':
            from geometry_match import ADDITIONAL_POSES,FRESH_TEST_FACES
            roles.update({p:[] for p in ADDITIONAL_POSES});fit_ids+=FRESH_TEST_FACES
        protocol.update(version=5 if args.calibration_set=='extended' else 4,
                        calibration_set=args.calibration_set,roles=roles,
                        missing_auxiliary_policy='skip absent auxiliary examples and record exact per-face calibration set',
                        contact_reference_policy='geometry-selected same-fit closed pose' if args.calibration_set=='extended' else 'PP',
                        fresh_test_faces=FRESH_TEST_FACES if args.calibration_set=='extended' else [],
                        assignment='geometry-matched source rig poses',
                        development_faces=DEVELOPMENT_FACES,new_test_faces=NEW_TEST_FACES,
                        face_ids=fit_ids,source_coefficient_regularization=.001,
                        evaluation_note='Fixed source queries derived only from development faces; no held-out target mesh used to construct its own query')
    if args.registration=='curves':
        protocol.update(version=6,source_registration='resampled anatomical source lip curves',
                        source_contact_labels='original nine verified rig pairs, independent of landmark registration')
    if args.robust:
        protocol.update(version=7,observation_policy='robust upper-face alignment, isolated contour anomaly repair, bounded closed-pose projection, point-wise Huber IRLS',
                        input_noise_test_required=True,raw_mesh_reconstruction_is_not_ground_truth=True)
    def load_face(path):
        data,selected,poses,frame,meta=face_data(path,args.robust)
        if args.robust:
            from robust_calibration import prepare_observations
            poses,confidence,notes=prepare_observations(poses)
            meta['_observation_repairs']=notes
            meta['_confidence']={k:v.tolist() for k,v in confidence.items()}
        return data,selected,poses,frame,meta
    def solve(ref,target,coeff,source_ref,prior,ridge=.03,observations=None):
        if args.robust:
            from robust_calibration import fit_robust_mapping
            return fit_robust_mapping(ref,target,coeff,source_ref,prior,ridge,observations)
        return fit_mapping(ref,target,coeff,source_ref,prior,ridge)
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Frozen protocol changed: use a new output directory')
    write_json(protocol_path,protocol)
    model=args.sandbox_assets/'compact-v4-validation/v4-candidate.onnx'
    if sha(model)!=MODEL_SHA:raise ValueError('Pinned model checksum mismatch')
    provenance=dict(model_path=str(model),model_sha256=sha(model),
                    avatar_meshes={i:sha(args.avatars/i/'face-meshes.json') for i in fit_ids},
                    source_files={p.name:sha(p) for p in args.source_rig.glob('*James*')},
                    face_task_sha256=sha(args.face_task),
                    scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                    observation='Earlier exploratory runs superseded: ownership/display fixes and reference-rest contact eligibility made explicit')
    write_json(out/'inputs.json',provenance)
    registration_path=out/'source-registration.npz'
    if not registration_path.exists():
        write_json(out/'registration.json',build_registration(args.source_rig,args.sandbox_assets,args.face_task,out,args.registration=='curves'))
    reg=np.load(registration_path)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    source_reference=session.run(['blendshapes'],{'pcm':np.zeros((1,16000),np.float32)})[0].reshape(60,52)[30]
    study=args.sandbox_assets/'compact-v4-confirmation'
    rows=json.loads((study/'manifest.json').read_text())
    # Verify cached coefficient provenance against one exact replay before reuse.
    sample=rows[0]
    audio=Path(sample['audio'])
    if sha(audio)!=sample['sha256_audio']:raise ValueError('Replay audio checksum mismatch')
    replay=infer_windows(session,np.fromfile(audio,'<f4'))
    cached=np.load(study/'predictions'/MODEL_SHA/(sample['sha256_audio']+'.npy'))
    parity=float(np.max(np.abs(replay-cached)))
    if parity>1e-5:raise ValueError(f'Cached output replay parity failed: {parity}')
    library,source_n,source_d=build_library(rows,study,reg,source_reference,out)
    if args.assignment=='geometry':
        from geometry_match import build_geometry_library
        (out/'legacy-source-library.json').write_text((out/'source-library.json').read_text())
        library=build_geometry_library(library,source_n,source_d,source_reference,args.avatars,
                                       load_face,roles,reg['channels'])
        write_json(out/'source-library.json',dict(reference=source_reference.tolist(),roles=library,
                    description='Geometry-matched authored source rig poses; source speech streams remain unmodified network predictions'))
    source_pose=source_n+np.einsum('nck,k->nc',source_d,source_reference)
    sf=local_frame(source_pose)
    prior=np.einsum('nck,cd->ndk',source_d[:,:2],sf[2])/sf[1]
    # Lower face ownership mask; top face and irises remain unchanged.
    sp=normalize(source_pose,sf)
    mask=np.clip((sp[:,1]+.45)/.25,0,1)*np.clip((1.5-np.abs(sp[:,0]))/.4,0,1)
    mask[468:]=0;prior*=mask[:,None,None]
    coeff=np.array([library[r]['coefficients'] for r in roles])
    channel_mask=np.array([str(n).startswith(('mouth','jaw')) or str(n)=='cheekPuff'
                           for n in reg['channels']])
    # Freeze unowned controls at reference for fitting; retain exact raw source
    # vectors in the library. The resulting target basis ignores those controls.
    fit_coeff=np.where(channel_mask,coeff,source_reference)
    prior*=channel_mask[None,None,:]
    coef_index={r:i for i,r in enumerate(roles)}
    # Separate real speech sequences: neither voice used to select calibration poses.
    eval_rows=[r for r in rows if r['voice'] in PROTOCOL['held_out_voices']]
    tracks=[]
    for row in eval_rows:
        w=np.load(study/'predictions'/MODEL_SHA/(row['sha256_audio']+'.npy'))
        tracks.append(w)
    results=[];tiles=[];inventory=[];sensitivity=[]
    names={a['id']:a.get('name',a['id']) for a in json.loads((args.avatars/'index.json').read_text())['avatars']}
    source_all=np.concatenate(tracks)
    src_geoms=source_n+np.einsum('nck,tk->tnc',source_d,source_all)
    # source_n stores y-down; source contact threshold is the rig's 1 mm criterion.
    source_seal=source_contact_mask(reg,source_all)
    reference_distance=np.linalg.norm((source_all-source_reference)*channel_mask,axis=1)
    eligible_seal=source_seal & (reference_distance>=.15)
    for face_id in fit_ids:
        path=args.avatars/face_id/'face-meshes.json'
        data,selected,poses,frame,align_rms=load_face(path)
        face_roles=[r for r in roles if r in poses]
        if not set(ROLES)<=set(face_roles):raise ValueError('Required core viseme missing: '+face_id)
        rows_for_face=[list(roles).index(r) for r in face_roles]
        face_coeff=coeff[rows_for_face];face_fit_coeff=fit_coeff[rows_for_face]
        ref=poses['neutral']; target=np.array([poses[r] for r in face_roles])
        confidence=np.array([align_rms.get('_confidence',{}).get(r,np.ones(len(ref))) for r in face_roles])
        start=time.perf_counter()
        mapping,stats=solve(ref,target,face_fit_coeff,source_reference,prior,observations=confidence)
        mapping*=mask[:,None,None]
        fit_ms=(time.perf_counter()-start)*1000
        predictions=evaluate(ref,mapping,face_coeff,source_reference)
        errors={r:lip_error(predictions[i],poses[r]) for i,r in enumerate(face_roles)}
        raw_observations=face_data(path,args.robust)[2] if args.robust else poses
        raw_errors={r:lip_error(predictions[i],raw_observations[r]) for i,r in enumerate(face_roles)}
        loo={}
        for i,role in enumerate(face_roles):
            keep=np.arange(len(face_roles))!=i
            m,_=solve(ref,target[keep],face_fit_coeff[keep],source_reference,prior,observations=confidence[keep])
            m*=mask[:,None,None]
            loo[role]=lip_error(evaluate(ref,m,face_coeff[i],source_reference),poses[role])
        # Separate shape error from forced contact; corrective success alone is not a pass.
        raw=evaluate(ref,mapping,source_all,source_reference)
        raw_gap=gaps(raw)
        contact_name='PP';contact_valid=None;contact_candidates=[]
        if args.calibration_set=='extended':
            from geometry_match import choose_contact_reference
            contact_name,contact_valid,contact_candidates=choose_contact_reference(poses)
        fixed=[];corrections=[]
        for shape,distance,seal in zip(raw,reference_distance,source_seal):
            # Smooth rest protection; a source seal similar to the reference may
            # remain ambiguous on a naturally parted target. Report that failure.
            movement=min(1,float(distance)/.15)
            strength=float(seal)*movement
            v,c=contact_correct(shape,poses[contact_name],strength)
            fixed.append(v);corrections.append(c)
        fixed=np.array(fixed);fixed_gap=gaps(fixed)
        pp=mouth_metrics(poses['PP'])
        valid_pp=pp['max_gap']<=.01 and pp['min_gap']>=-.005
        seal_pass_raw=float(np.mean(np.max(raw_gap[eligible_seal],axis=1)<=.01))
        seal_pass=float(np.mean(np.max(fixed_gap[eligible_seal],axis=1)<=.01))
        max_corr=float(max(corrections))
        if contact_valid is None:contact_valid=valid_pp
        result=dict(id=face_id,calibration_roles=face_roles,
                    missing_auxiliary_poses=[r for r in roles if r not in poses],name=names.get(face_id,face_id),fit_ms=fit_ms,svd=stats,
                    rest_max_error=float(np.max(np.abs(evaluate(ref,mapping,source_reference,source_reference)-ref))),
                    calibration_lip_rms=errors,raw_observation_lip_rms=raw_errors,
                    observation_repairs=align_rms.get('_observation_repairs',{}),
                    raw_PP_metrics=mouth_metrics(raw_observations['PP']),
                    leave_one_viseme_out_lip_rms=loo,
                    PP_metrics=pp,PP_valid_contact_example=valid_pp,
                    contact_reference=contact_name,contact_reference_valid=contact_valid,
                    contact_reference_candidates=contact_candidates,
                    heldout_speech=dict(frames=len(source_all),source_sealed_frames=int(source_seal.sum()),
                        eligible_active_source_sealed_frames=int(eligible_seal.sum()),
                        reference_like_source_sealed_frames=int(np.sum(source_seal & ~eligible_seal)),
                        raw_seal_pass_fraction=seal_pass_raw,corrected_seal_pass_fraction=seal_pass,
                        raw_penetrating_frame_fraction=float(np.mean(np.min(raw_gap,axis=1)<-.005)),
                        corrected_penetrating_frame_fraction=float(np.mean(np.min(fixed_gap,axis=1)<-.005)),
                        max_correction_mouth_widths=max_corr),
                    calibration_pass=max(errors.values())<=.03,
                    heldout_viseme_pass=max(loo.values())<=.06,
                    speech_contact_pass=seal_pass>=.99 and max_corr<=.05,
                    visual_review_required=True)
        result['numerical_screen_pass']=all([contact_valid,result['calibration_pass'],result['heldout_viseme_pass'],result['speech_contact_pass']])
        results.append(result)
        for ridge in PROTOCOL['sensitivity_ridges']:
            m,_=solve(ref,target,face_fit_coeff,source_reference,prior,ridge,observations=confidence)
            m*=mask[:,None,None]
            train=evaluate(ref,m,face_coeff,source_reference)
            held=[]
            for i in range(len(face_roles)):
                keep=np.arange(len(face_roles))!=i
                h,_=solve(ref,target[keep],face_fit_coeff[keep],source_reference,prior,ridge,observations=confidence[keep])
                h*=mask[:,None,None]
                held.append(lip_error(evaluate(ref,h,face_coeff[i],source_reference),target[i]))
            sensitivity.append(dict(face=face_id,ridge=ridge,
                max_calibration_lip_rms=max(lip_error(a,b) for a,b in zip(train,target)),
                max_leave_one_out_lip_rms=max(held),exploratory=True))
        inventory.append(dict(id=face_id,visemes={r:mouth_metrics(poses[r]) for r in ['neutral',*face_roles]},alignment_rms=align_rms))
        np.savez_compressed(out/(face_id+'-mapping.npz'),reference=ref,basis=mapping,
                            source_reference=source_reference,coefficients=face_coeff,
                            frame_center=frame[0],frame_width=frame[1],frame_rotation=frame[2])
        for r in ['PP','aa','ou']:
            v=selected[r];raw_lm=np.array(v['landmarks'])
            img=Image.open(args.avatars/face_id/v['sourceFile'])
            tiles.append(annotated(img,raw_lm,predictions[face_roles.index(r)],frame,
                         f'{face_id} {r}: fitted green / mapped red',poses[r]))
        print(face_id,'cal',round(max(errors.values()),3),'LOO',round(max(loo.values()),3),
              'PP',valid_pp,'speech contact',round(seal_pass,3),'max correction',round(max_corr,3),flush=True)
    sheet=Image.new('RGB',(440*3,320*len(fit_ids)),'#10141c')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*440,(i//3)*320))
    sheet.save(out/'viseme-fit-comparison.png')
    write_json(out/'inventory.json',inventory)
    write_json(out/'regularization-sensitivity.json',sensitivity)
    write_json(out/'results.json',dict(protocol=protocol,cache_replay_max_abs_error=parity,
                 heldout_voices=PROTOCOL['held_out_voices'],heldout_clips=len(eval_rows),faces=results,
                 numerical_screen_pass=all(r['numerical_screen_pass'] for r in results),
                 conclusion='No player integration until mapping and visual gates pass.'))


if __name__=='__main__':main()

"""Select a local residual on development faces, then evaluate other actual fits."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from run_probe import face_data,source_xy,write_json,sha,annotated
from mapping import local_frame,normalize,lip_error,contact_correct,gaps,mouth_metrics
from geometry_match import DEVELOPMENT_FACES,source_feature_basis,choose_contact_reference
from robust_calibration import prepare_observations
from local_mapping import fit_local,evaluate_local,adapt_lip_prior
from source_registration import source_contact_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--avatars',type=Path,required=True);p.add_argument('--sandbox-assets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=args.input;out=args.out
    out.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((root/'results.json').read_text());library=json.loads((root/'source-library.json').read_text())
    protocol=dict(version=1,development_faces=DEVELOPMENT_FACES,
        bandwidth_factors=[.5,1.,2.],ridges=[.0001,.001,.01],
        calibration_limit=.03,heldout_viseme_limit=.06,contact_gap_limit=.01,
        max_runtime_correction=.05,heldout_clips=48,
        selection='Minimize worst development leave-one-pose-out RMS among settings meeting the 3% calibration limit; no evaluation face used in selection',
        observation_policy='Noisy observations; source-curves correspondence, robust anchor alignment, isolated-point repair, bounded closed-pose repair, anatomical prior',
        scope='Research only; passing numerical checks does not authorize player changes')
    pp=out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Protocol changed')
    write_json(pp,protocol)
    for name in ['source-library.json','source-registration.npz','source-registration-overlay.png','inputs.json']:
        import shutil
        shutil.copy2(root/name,out/name)
    reg=np.load(root/'source-registration.npz');n,b=source_xy(reg);c=np.array(library['reference'])
    owned=np.array([str(x).startswith(('mouth','jaw')) or str(x)=='cheekPuff' for x in reg['channels']])
    source=n+np.einsum('nck,k->nc',b,c);sf=local_frame(source)
    source_ref_geometry=normalize(source,sf)
    base_prior=np.einsum('nck,cd->ndk',b[:,:2],sf[2])/sf[1]
    sp=source_ref_geometry;mask=np.clip((sp[:,1]+.45)/.25,0,1)*np.clip((1.5-np.abs(sp[:,0]))/.4,0,1);mask[468:]=0
    base_prior*=mask[:,None,None]*owned[None,None,:]
    features=source_feature_basis(n,b,c,owned)
    faces={}
    for f in baseline['faces']:
        face=f['id'];data,selected,raw,frame,_=face_data(args.avatars/face/'face-meshes.json',True)
        poses,confidence,notes=prepare_observations(raw)
        keys=f['calibration_roles'];coeff=np.array([library['roles'][k]['coefficients'] for k in keys]);coeff=np.where(owned,coeff,c)
        prior,ratios=adapt_lip_prior(poses['neutral'],source_ref_geometry,base_prior)
        faces[face]=dict(data=data,selected=selected,raw=raw,poses=poses,frame=frame,keys=keys,
            coeff=coeff,target=np.array([poses[k] for k in keys]),confidence=np.array([confidence[k] for k in keys]),
            notes=notes,prior=prior,ratios=ratios)
    def build(f,factor,ridge,keep=None):
        if keep is None:keep=np.ones(len(f['keys']),bool)
        return fit_local(f['poses']['neutral'],f['target'][keep],f['coeff'][keep],c,f['prior'],features,
                         factor,ridge,f['confidence'][keep],mask)
    def pose_checks(f,factor,ridge):
        model=build(f,factor,ridge);pred=evaluate_local(model,f['coeff'])
        cal={k:lip_error(pred[i],f['target'][i]) for i,k in enumerate(f['keys'])}
        held={}
        for i,key in enumerate(f['keys']):
            keep=np.arange(len(f['keys']))!=i
            m=build(f,factor,ridge,keep)
            held[key]=lip_error(evaluate_local(m,f['coeff'][i]),f['target'][i])
        return model,pred,cal,held
    selection=[]
    for factor in protocol['bandwidth_factors']:
        for ridge in protocol['ridges']:
            rows=[]
            for face in DEVELOPMENT_FACES:
                _,_,cal,held=pose_checks(faces[face],factor,ridge)
                rows.append(dict(face=face,max_calibration=max(cal.values()),max_heldout=max(held.values())))
            row=dict(factor=factor,ridge=ridge,faces=rows,max_calibration=max(x['max_calibration'] for x in rows),
                     worst_development_heldout=max(x['max_heldout'] for x in rows))
            selection.append(row)
            print('development',factor,ridge,'cal',round(row['max_calibration'],3),'held',round(row['worst_development_heldout'],3),flush=True)
    eligible=[x for x in selection if x['max_calibration']<=.03]
    chosen=min(eligible or selection,key=lambda x:(x['worst_development_heldout'],x['max_calibration']))
    write_json(out/'development-selection.json',dict(grid=selection,chosen=chosen,
                meets_development_calibration_limit=bool(eligible),evaluation_faces_used=False))
    study=args.sandbox_assets/'compact-v4-confirmation';rows=[x for x in json.loads((study/'manifest.json').read_text()) if x['voice'] in ['M3','F3']]
    tracks=np.concatenate([np.load(study/'predictions'/baseline['protocol']['model_sha256']/(x['sha256_audio']+'.npy')) for x in rows])
    seal=source_contact_mask(reg,tracks);distance=np.linalg.norm((tracks-c)*owned,axis=1);eligible_seal=seal&(distance>=.15)
    results=[];tiles=[]
    for face,f in faces.items():
        start=time.perf_counter();model,pred,cal,held=pose_checks(f,chosen['factor'],chosen['ridge'])
        fit_checks_ms=(time.perf_counter()-start)*1000
        raw=evaluate_local(model,tracks)
        contact,valid,candidates=choose_contact_reference(f['poses']);fixed=[];corrections=[]
        for shape,is_seal,dist in zip(raw,seal,distance):
            v,amount=contact_correct(shape,f['poses'][contact],float(is_seal)*min(1,float(dist)/.15));fixed.append(v);corrections.append(amount)
        fixed=np.array(fixed);pass_fraction=float(np.mean(np.max(gaps(fixed[eligible_seal]),axis=-1)<=.01))
        result=dict(id=face,development=face in DEVELOPMENT_FACES,calibration_roles=f['keys'],
            calibration_lip_rms=cal,leave_one_viseme_out_lip_rms=held,
            raw_observation_lip_rms={k:lip_error(pred[i],f['raw'][k]) for i,k in enumerate(f['keys'])},
            observation_repairs=f['notes'],anatomical_prior=f['ratios'],
            contact_reference=contact,contact_reference_valid=valid,contact_candidates=candidates,
            corrected_seal_pass_fraction=pass_fraction,max_runtime_contact_correction=float(max(corrections)),
            fit_and_all_pose_checks_ms=fit_checks_ms,reference_error=float(np.max(np.abs(evaluate_local(model,c)-f['poses']['neutral']))))
        result['numerical_screen_pass']=bool(max(cal.values())<=.03 and max(held.values())<=.06 and valid and pass_fraction>=.99 and max(corrections)<=.05)
        results.append(result);np.savez_compressed(out/(face+'-local-map.npz'),**model)
        for key in ['PP','aa','ou']:
            v=f['selected'][key];img=Image.open(args.avatars/face/v['sourceFile'])
            tiles.append(annotated(img,np.array(v['landmarks']),pred[f['keys'].index(key)],f['frame'],
                         face+' '+key+': observed green / corrected red',f['poses'][key]))
        print(face,'cal',round(max(cal.values()),3),'held',round(max(held.values()),3),'pass',result['numerical_screen_pass'],flush=True)
    sheet=Image.new('RGB',(1320,320*len(results)),'#151921')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*440,(i//3)*320))
    sheet.save(out/'viseme-fit-comparison.png')
    write_json(out/'results.json',dict(protocol=protocol,model_sha256=baseline['protocol']['model_sha256'],
        selected_parameters=dict(factor=chosen['factor'],ridge=chosen['ridge']),frames=len(tracks),faces=results,
        numerical_screen_pass=all(f['numerical_screen_pass'] for f in results),player_changes=False))


if __name__=='__main__':main()

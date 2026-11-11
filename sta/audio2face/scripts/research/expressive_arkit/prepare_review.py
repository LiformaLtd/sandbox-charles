"""Stage a separate full-52 James rig and small audio set for browser review."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import sys
import numpy as np
from scipy.io import wavfile
from common import *
from fit import FullFaceFit,safe_offset,mm
from gaze import NEW


def jaw_delta(neutral,delta,anchors):
    target=neutral+delta;center=neutral.mean(0);moved=target.mean(0)
    u,_,v=np.linalg.svd(mm((neutral-center).T,target-moved))
    u[:,-1]*=np.linalg.det(mm(u,v));rotation=mm(u,v)
    return mm(anchors-center,rotation)+moved-anchors


def main():
    parser=argparse.ArgumentParser()
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--verified-candidate',type=Path,help='Directory containing model.onnx and passed verification.json')
    selection.add_argument('--review-candidate',type=Path,help='Stage for visual review with known regression failures displayed')
    parser.add_argument('--extra-sample',action='append',default=[],help='Additional recording stem from the research manifest')
    args=parser.parse_args()
    candidate=NEW/'fullface.onnx';verification=None
    candidate_dir=args.verified_candidate or args.review_candidate
    if candidate_dir:
        candidate=candidate_dir/'model.onnx'
        verification=json.loads((candidate_dir/'verification.json').read_text())
        if verification['model_sha256']!=sha(candidate):
            raise ValueError('Verification model SHA does not match candidate')
        if not verification['passed'] and not args.review_candidate:
            raise ValueError('Candidate must pass verification with matching model SHA')
    candidate_name='fullface-'+sha(candidate)[:16]+'.onnx'
    out=APP/'public/assets/expressive';out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(APP/'scripts'))
    import prepare_blendshapes as pack
    from prepare_rig import maya_nodes
    pack.offset_transform=safe_offset
    fit=FullFaceFit(ASSETS,WORK/'arkit-student/james_arkit_v3.ma',APP)
    nodes=maya_nodes((WORK/'arkit-student/james_arkit_v3.ma').read_text())
    teeth_neutral,teeth=pack.maya_deltas(nodes,'c_bottomDenture_hi_bs','c_bottomDenture_hiShapeOrig',fit.names)
    with np.load(ASSETS/'model_data_James.npz') as m:jaw={n:jaw_delta(teeth_neutral,d,m['neutral_jaw']) for n,d in teeth.items()}
    skin={n:fit.shape[i] for i,n in enumerate(fit.names)}
    skin['neutralOffset']=fit.neutral-fit.model_neutral
    tongue={n:fit.tongue_mouth[i] for i,n in enumerate(fit.names)}
    tongue['tongueOut']=fit.tongue_basis[fit.tongue_out_index].reshape(-1,3)
    tongue['neutralOffset']=fit.tongue_neutral-fit.model_tongue
    with (out/'full-blendshapes.bin').open('wb') as f:
        f.write(b'JBS1')
        for group in [skin,tongue,jaw]:pack.pack_group(f,group,fit.names,list(range(52)))
    inputs={}
    for src,name in [(SOURCE_APP/'.local/rig.json','rig.json'),(SOURCE_APP/'.local/rig.bin','rig.bin'),
        (SOURCE_APP/'.local/arkit-student/james-blendshapes.bin','baseline-blendshapes.bin'),
        (BASELINE,'baseline.onnx'),(candidate,candidate_name)]:
        shutil.copyfile(src,out/name);inputs[name]=dict(source=str(src),sha256=sha(src))
    appointment=Path('/Users/charleswiles/Source/liforma/sandboxes/sandbox-charles/sta/audio2face-2d/static/assets/speech/appointment-supertonic-m1.wav')
    rate,pcm=wavfile.read(appointment)
    if rate!=16000 or pcm.dtype!=np.int16 or pcm.ndim!=1:raise ValueError('Unexpected appointment fixture')
    pcm=pcm.astype(np.float32)/32768
    rows=json.loads((OUT/'manifest.json').read_text())
    selected=[r for r in rows if r['split']=='test' and r['stem'].endswith('-000') and r['voice'] in ['Hugh','F5']]
    for stem in args.extra_sample:
        matches=[r for r in rows if r['stem']==stem]
        if len(matches)!=1:raise ValueError('Unknown or duplicate recording stem: '+stem)
        if all(r['stem']!=stem for r in selected):selected.append(matches[0])
    cases=[dict(id='appointment',label='Appointment · exact regression',text='Good afternoon, I am here to help you arrange your next appointment.',pcm=pcm,source=str(appointment))]
    for r in selected:
        if sha(r['audio'])!=r['sha256_audio']:raise ValueError('Changed source recording: '+r['stem'])
        label=r['engine'].title()+' · '+r['voice']
        if r['stem'] in args.extra_sample:label+=' · neutral brow example ('+r['split']+')'
        cases.append(dict(id=r['stem'],label=label,text=r['text'],pcm=np.fromfile(r['audio'],'<f4'),source=r['audio']))
    samples=[]
    for case in cases:
        f32=out/(case['id']+'.f32');wav=out/(case['id']+'.wav')
        case['pcm'].astype('<f4').tofile(f32);wavfile.write(wav,16000,case['pcm'])
        samples.append(dict(id=case['id'],label=case['label'],text=case['text'],
            pcm='/assets/expressive/'+f32.name,wav='/assets/expressive/'+wav.name,
            samples=len(case['pcm']),sha256=sha(f32),source=case['source']))
    (NEW/'review-provenance.json').write_text(json.dumps(dict(inputs=inputs,samples=samples),indent=2))
    public_samples=[{k:v for k,v in sample.items() if k!='source'} for sample in samples]
    lip_reference=None
    if verification:
        with np.load(candidate_dir/'raw-gap-references.npz') as z:
            references={k:z[k].tolist() for k in z.files}
        with np.load(OUT/'calibration.npz') as z:
            bank=dict(basis=z['gap_basis'].tolist(),rest=z['gap_neutral'].tolist(),references=references,
                limits={r['reference_key']:r['old_lip_rms_mm']*verification['lip_tolerance'] for r in verification['phrases']})
        name='lip-reference-'+sha(candidate)[:16]+'.json'
        (out/name).write_text(json.dumps(bank))
        lip_reference=dict(url='/assets/expressive/'+name,sha256=sha(out/name))
    manifest=dict(channels=fit.names,emotions=EMOTIONS,
        gazeStyles=['Claire','James','Mark'],samples=public_samples,
        baseline=dict(url='/assets/expressive/baseline.onnx',sha256=sha(BASELINE)),
        candidate=dict(url='/assets/expressive/'+candidate_name,sha256=sha(candidate),
            reviewOnly=bool(args.review_candidate),
            violations=verification['violations'] if verification else [],
            reviewWarnings=verification.get('review_warnings',[]) if verification else [],
            reviewNotes=verification.get('review_notes',[]) if verification else [],
            reviewResults=[dict(sample=r['sample'],emotion=r['emotion'],changePercent=100*(r['lip_ratio']-1))
                for r in verification['phrases'] if r['lip_ratio']>verification['lip_tolerance']] if verification else [],
            label=candidate_dir.name if candidate_dir else 'Original full-face candidate'),
        lipReference=lip_reference,headMotion=False)
    if (out/'review.json').exists():
        previous=json.loads((out/'review.json').read_text())
        references=previous.get('teacherReviews',{}).copy()
        if previous.get('teacherReview'):references[previous['teacherReview']['sample']]=previous['teacherReview']
        audio_hashes={s['id']:s['sha256'] for s in public_samples}
        kept={}
        for key,reference in references.items():
            if audio_hashes.get(reference['sample'])!=reference['audioSha256']:continue
            track=APP/'public'/reference['url'].lstrip('/')
            if sha(track)!=reference['sha256']:raise ValueError('Changed teacher replay: '+key)
            kept[key]=reference
        if kept:
            manifest['teacherReviews']=kept
            default=previous.get('teacherReview',{})
            manifest['teacherReview']=next((r for r in kept.values() if r['sample']==default.get('sample') and r['emotion']==default.get('emotion')),next(iter(kept.values())))
    temporary=out/'review.json.tmp'
    temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(out/'review.json')
    print('Prepared',out,'samples',len(samples),'full 52 targets; no head motion')


if __name__=='__main__':main()

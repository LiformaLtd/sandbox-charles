"""Package the geometry matching and noisy-observation research results."""
import argparse,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw
from run_probe import face_data,annotated,source_xy,write_json
from mapping import local_frame,normalize
from safety_check import lip_thickness
from robust_calibration import prepare_observations


def main():
    p=argparse.ArgumentParser();p.add_argument('--research-root',type=Path,required=True)
    p.add_argument('--avatars',type=Path,required=True);p.add_argument('--sandbox-assets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=args.research_root;out=args.out
    out.mkdir(parents=True,exist_ok=True)
    paths=dict(original=root/'mediapipe-mapping-pilot-v3',
        geometry=root/'mediapipe-mapping-geometry-core-v2',extended=root/'mediapipe-mapping-geometry-extended-v2',
        robust=root/'mediapipe-mapping-robust-curves-v1',local=root/'mediapipe-mapping-local-v1')
    data={k:json.loads((v/'results.json').read_text()) for k,v in paths.items()}
    noise=json.loads((paths['robust']/'noise-results.json').read_text())
    safety=json.loads((paths['robust']/'geometry-safety.json').read_text())
    dev=set(data['geometry']['protocol']['development_faces'])
    old={f['id']:f for f in data['original']['faces']};new={f['id']:f for f in data['geometry']['faces']}
    common=[i for i in old if i not in dev]
    comparison={}
    for label,key in [('calibration','calibration_lip_rms'),('withheld_viseme','leave_one_viseme_out_lip_rms')]:
        a=float(np.mean([max(old[i][key].values()) for i in common]));b=float(np.mean([max(new[i][key].values()) for i in common]))
        comparison[label]=dict(old=a,geometry_matched=b,reduction_fraction=1-b/a)
    x=np.arange(len(common));fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    ax.bar(x-.18,[max(old[i]['leave_one_viseme_out_lip_rms'].values())*100 for i in common],.36,label='Original source assignment')
    ax.bar(x+.18,[max(new[i]['leave_one_viseme_out_lip_rms'].values())*100 for i in common],.36,label='Geometry source assignment')
    ax.axhline(6,color='#a23a34',linestyle='--',label='6% withheld-pose screen')
    ax.set_xticks(x,common,rotation=30,ha='right');ax.set_ylabel('Worst withheld-pose lip RMS / mouth width (%)')
    ax.set_title('Controlled assignment-only comparison on seven common evaluation fits')
    ax.legend(fontsize=8);fig.savefig(out/'geometry-matching-comparison.png',dpi=160);plt.close(fig)
    faces=data['robust']['faces'];byid={f['id']:f for f in faces};noise_byid={f['id']:f for f in noise['faces']}
    repaired_ids=['dd80126a','394d2cb9','c1bb51a7','d3194283','64cf2cc1','201c8934']
    tiles=[]
    for face in repaired_ids:
        _,selected,raw,frame,_=face_data(args.avatars/face/'face-meshes.json',True)
        latent,_,_=prepare_observations(raw);v=selected['PP']
        image=Image.open(args.avatars/face/v['sourceFile'])
        label=f"{face}: observed green / closed-pose repair red"
        tiles.append(annotated(image,np.array(v['landmarks']),latent['PP'],frame,label,latent['PP']))
    sheet=Image.new('RGB',(1320,640),'#151921')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*440,(i//3)*320))
    sheet.save(out/'closed-mouth-repairs.png')
    source_sheet=Image.new('RGB',(1300,690),'#151921')
    for i,(label,path) in enumerate([
        ('Previous probe registration',paths['original']/'source-registration-overlay.png'),
        ('Corrected anatomical curve registration',paths['robust']/'source-registration-overlay.png')]):
        source_sheet.paste(Image.open(path).resize((650,650)),(i*650,40))
        ImageDraw.Draw(source_sheet).text((i*650+12,12),label,fill='white')
    source_sheet.save(out/'source-registration-comparison.png')
    study=args.sandbox_assets/'compact-v4-confirmation';rows=[r for r in json.loads((study/'manifest.json').read_text()) if r['voice'] in ['M3','F3']]
    tracks=np.concatenate([np.load(study/'predictions'/data['robust']['protocol']['model_sha256']/(r['sha256_audio']+'.npy')) for r in rows])
    reference=np.array(json.loads((paths['robust']/'source-library.json').read_text())['reference'])
    registration_stats={}
    for name,path in [('previous',paths['original']),('corrected',paths['robust'])]:
        n,b=source_xy(np.load(path/'source-registration.npz'));source=n+np.einsum('nck,k->nc',b,reference)
        points=normalize(n+np.einsum('nck,tk->tnc',b,tracks),local_frame(source))
        thickness=lip_thickness(points)
        registration_stats[name]=dict(negative_thickness_frame_fraction=float(np.mean(np.min(thickness,axis=1)<-.005)),minimum_thickness=float(thickness.min()))
    noise_stats={}
    for case in ['lip_vertex_spike','false_closed_gap','gaussian_lip_noise']:
        values=[next(t for t in f['corruption_tests'] if t['case']==case)['methods'] for f in noise['faces']]
        a=np.array([x['ordinary']['lip_rms'] for x in values]);b=np.array([x['robust']['lip_rms'] for x in values])
        noise_stats[case]=dict(median_ordinary_rms=float(np.median(a)),median_robust_rms=float(np.median(b)),
            median_relative_reduction=float(np.median(1-b/a)),worst_robust_rms=float(b.max()),
            worst_robust_point_change=max(x['robust']['maximum_point_change'] for x in values))
    fit_times=[f['fit_ms'] for f in faces]
    write_json(out/'comparison-summary.json',dict(assignment_only=comparison,source_registration=registration_stats,
        noise=noise_stats,pointwise_warning='Mean RMS stability is not a pointwise robustness guarantee',player_changes=False))
    table=[]
    for f in faces:
        table.append(f"| {f['id']} | {f['raw_PP_metrics']['max_gap']*100:.2f}% | {f['PP_metrics']['max_gap']*100:.2f}% | {max(f['leave_one_viseme_out_lip_rms'].values())*100:.1f}% |")
    report=f'''# Geometry matched mapping and noisy mesh calibration results

Date: 2026-11-07. **Research improved, but the mapping is not cleared for v1 integration. The player and original avatar assets remain unchanged.**

## Goal and quality standard

The objective is coherent speech motion on each face, not exact reproduction of every old mesh. Existing viseme fits are noisy observations. Clear fitting defects can be corrected when the correction is supported by anatomy, neighboring landmarks or consistent poses. Face proportions, asymmetry and naturally parted rest must remain recognizable. A visually plausible but wholly incorrect fit cannot be identified reliably from geometry alone; it needs image evidence or a trusted independent observation.

Reconstruction error against raw meshes is therefore reported separately from error against anatomically constrained calibration targets. Matching an erroneous raw PP gap is not treated as success. Conversely, neither a small mean error nor a forced lip seal is enough to approve a mapping.

## Work completed

- Replaced phoneme-role representative source frames with bounded source-rig coefficient fits matched to actual mouth geometry: detailed lip displacement, aperture profile, spread and upper/lower lip motion. Source assignments use four development faces only. Eye and brow controls remain excluded from speech geometry.
- Tested the same original mesh fits, then expanded the stress set to **19 fits**. Up to 22 existing viseme/expression examples are used per fit; absent auxiliary examples are recorded, not silently substituted. Three versions of one avatar are included.
- Corrected an error in the probe's source correspondence. The nine verified source lip-pair samples cover a central portion of the mouth; assigning them directly to nine MediaPipe indices compressed the inner contour. The new registration resamples the source lip curves at anatomical locations across the full mouth. The source model itself was not changed.
- Added robust upper-face alignment, isolated contour anomaly handling, bounded closed-pose corrections and point-wise robust fitting. Original photographs, UV/source landmarks and asset files are never overwritten. Statistical contour repairs remain proposals requiring visual review, especially on stylized faces.
- Tested injected errors using in-memory copies of actual fits: a lip landmark displaced by 25% of mouth width, an artificial 8% closed-mouth gap, 1% Gaussian lip noise and one damaged alignment anchor.
- Tested a local nonlinear residual with a fit-specific anatomical prior. Its parameters were selected using only the four development faces before the remaining evaluation fits were processed. It did not solve the withheld-pose problem and is not accepted as a production replacement.

All speech checks reuse the same 48 M3/F3 clips ({len(tracks):,} frames) held out of mapping-example selection and the originally pinned compact ONNX export. Source seal labels still come from the original nine verified source rig pairs, so correcting registration does not change that label definition. The test sets have been reused across exploratory iterations; these are engineering comparisons, not a blinded final quality trial.

The robust residual solve took {min(fit_times):.1f}–{max(fit_times):.1f} ms per face on this Mac, compared with less than 1 ms for the original simple linear solve. This excludes asset loading, model startup and browser/mobile overhead. Robust calibration adds work, but it is still a once-per-fit operation; these figures are not a browser latency promise.

## Geometry matching improvement

In a controlled assignment-only comparison using the same seven core poses, solver and previous registration, the seven common evaluation fits outside the development set showed:

- Mean worst calibration error: {comparison['calibration']['old']*100:.2f}% → {comparison['calibration']['geometry_matched']*100:.2f}% of neutral mouth width, a {comparison['calibration']['reduction_fraction']*100:.0f}% reduction.
- Mean worst withheld-viseme error: {comparison['withheld_viseme']['old']*100:.2f}% → {comparison['withheld_viseme']['geometry_matched']*100:.2f}%, a {comparison['withheld_viseme']['reduction_fraction']*100:.0f}% reduction. This remains above the 6% screen.

This isolates the benefit of improved source assignment. It is not the final quality score of the later robust pipeline, which changes both correspondence and observation treatment.

![Controlled geometry matching comparison](geometry-matching-comparison.png)

## Source correspondence correction

On the same held-out coefficients, the previous source registration inverted the projected inner/outer lip thickness ordering on {registration_stats['previous']['negative_thickness_frame_fraction']*100:.1f}% of source frames. The corrected registration reduced this diagnostic to {registration_stats['corrected']['negative_thickness_frame_fraction']*100:.1f}%. This fixes an error in the research mapping rather than blaming it on an avatar or the audio model. Full 3D correspondence outside the mouth still needs review.

![Previous and corrected source lip correspondence](source-registration-comparison.png)

## Correcting existing closed-mouth fitting defects

The calibration now estimates a closed contact shape instead of requiring an imperfect PP mesh to be an exact target. It preserves the fitted lip midline and moves the upper/lower boundaries together with their outer rims, retaining thickness rather than squeezing lips into a line. Repairs are bounded and recorded separately from runtime deformation. Parted neutral mouths are not forced closed.

For the largest observed PP gap, the measured aperture was reduced from approximately 6.2% to 0.3% of mouth width with about 3% movement per lip. This is a concrete geometric repair in the calibration target, **not yet proof of better complete textured speech animation**.

Green shows original fitted contours; red shows the proposed closed-pose calibration repair. The photographs and textures are unchanged and are not warped in this comparison.

![Examples of closed-mouth calibration repairs](closed-mouth-repairs.png)

| Actual fit | Raw PP maximum gap | Repaired calibration gap | Worst withheld-pose RMS in robust linear probe |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}

The last column is an engineering diagnostic against constrained observations, not a declaration that every withheld old mesh is ground truth. Remaining failures also include motion-induced thickness loss and large localized responses, so raw reconstruction scores are not the sole reason integration remains blocked.

## Robustness measurements

The median influence of an injected isolated lip-vertex error fell by approximately {noise_stats['lip_vertex_spike']['median_relative_reduction']*100:.0f}%. Added false closed-mouth gaps were largely removed by the contact constraint. Damaging one alignment anchor no longer moves the entire mouth substantially. All 19 fits stayed within the 3% average lip-RMS stability screen for the chosen corruption cases.

However, average stability hides localized failures: the worst individual point still changed by about {max(v['worst_robust_point_change'] for v in noise_stats.values())*100:.0f}% of mouth width in a stress case. Gaussian noise was not improved on average. The robust calibration therefore **has not passed a complete robustness gate**. Strong smoothing or rejecting a distinctive whole pose would not be an acceptable way to hide this failure.

The image-plane triangle-orientation check is deliberately conservative. Some original valid-looking fits already reverse tiny projected triangles relative to neutral, so that count alone is not proof of a rendering defect. It is retained alongside the source-asset baseline and lip-thickness results for review. The actual production depth/oral/blink compositor has not been modified or used to claim a pass.

## Nonlinear trial and remaining work

The local Gaussian residual fits calibration examples almost exactly and returns toward an anatomical prior outside the calibrated region. It still misses the 6% withheld-pose screen across the 19-fit set. More fitting flexibility alone is therefore insufficient.

The next iteration should establish image-grounded confidence for questionable observations and validate cross-pose source/target correspondence, especially rounded vowels and stylized mouths. Keep bounded contact and thickness constraints, add pointwise stability limits, and verify temporal continuity. Use reviewed motion/appearance comparisons as well as geometry diagnostics. A mapper should improve a clear defect, retain genuine facial character and flag unsupported ambiguity; it should not blindly copy bad fits or silently invent a different face.

No player refactor, selector wiring, model deployment or default switch should proceed from these results. Browser calibration remains the initial implementation location after verification; publish-time mapping remains the agreed later architecture.

## Reproduction and preservation

The research branch is `research/mediapipe-mapping-feasibility` in its dedicated sandbox worktree. The archive contains scripts, pinned requirements, protocols, results and run instructions. Numerical limits were not relaxed to obtain a pass. Sixteen invariant tests pass; cached coefficients retain the original exact replay check. Original mesh and model hashes and the complete Player Git diff are checked before delivery.

The primary Player checkout contained unrelated location UI changes at the start of this work. They are preserved. No v1 player code, original avatar source mesh, TTS endpoint, CDN publisher or deployment was changed.
'''
    (out/'mapping-improvement-results.md').write_text(report)
    for key,path in paths.items():
        write_json(out/(key+'-results.json'),data[key])
        if (path/'protocol.json').exists():shutil.copy2(path/'protocol.json',out/(key+'-protocol.json'))
    for name in ['noise-results.json','noise-protocol.json','geometry-safety.json','safety-protocol.json','source-library.json','inputs.json']:
        shutil.copy2(paths['robust']/name,out/name)
    shutil.copy2(paths['local']/'development-selection.json',out/'local-development-selection.json')
    (out/'source-registration-diagnostics.json').write_text(json.dumps(registration_stats,indent=2)+'\n')
    docs=Path(__file__).resolve().parents[3]/'docs/plans';docs.mkdir(parents=True,exist_ok=True)
    local_assets=root/'mediapipe-mapping-improvement-report';local_assets.mkdir(exist_ok=True)
    local_report=report
    for name in ['geometry-matching-comparison.png','source-registration-comparison.png','closed-mouth-repairs.png']:
        shutil.copy2(out/name,local_assets/name)
        local_report=local_report.replace(']('+name+')','](../../.local/mediapipe-mapping-improvement-report/'+name+')')
    (docs/'mediapipe-mapping-improvement-results.md').write_text(local_report)
    print(out/'mapping-improvement-results.md')


if __name__=='__main__':main()

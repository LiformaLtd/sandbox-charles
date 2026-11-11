"""Create the inspectable feasibility report, plots and source-pose contact sheet."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from source_registration import render_source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=args.run;out=args.output;out.mkdir(parents=True,exist_ok=True)
    d=json.loads((r/'results.json').read_text());faces=d['faces']
    sensitivity=json.loads((r/'regularization-sensitivity.json').read_text())
    library=json.loads((r/'source-library.json').read_text())
    reg=np.load(r/'source-registration.npz')
    source_sheet=Image.new('RGB',(400*4,310*2),'#151921')
    for i,(role,pose) in enumerate(library['roles'].items()):
        points=reg['source_neutral']+np.einsum('nck,k->nc',reg['source_basis'],pose['coefficients'])
        path=r/('source-pose-'+role+'.png')
        render_source(points,reg['source_triangles'],path)
        crop=Image.open(path).crop((245,525,655,795)).resize((400,265))
        tile=Image.new('RGB',(400,310),'#151921');tile.paste(crop,(0,45))
        ImageDraw.Draw(tile).text((10,12),f'{role}: observed model pose, not a pure channel',fill='white')
        source_sheet.paste(tile,((i%4)*400,(i//4)*310))
    source_sheet.save(out/'source-calibration-poses.png')
    ids=[f['id'] for f in faces]
    train=[max(f['calibration_lip_rms'].values())*100 for f in faces]
    held=[max(f['leave_one_viseme_out_lip_rms'].values())*100 for f in faces]
    x=np.arange(len(ids))
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    ax.bar(x-.18,train,.36,label='Worst calibration pose RMS')
    ax.bar(x+.18,held,.36,label='Worst leave-one-viseme-out RMS')
    ax.axhline(3,color='#346b27',linestyle='--',label='Calibration limit 3%')
    ax.axhline(6,color='#a34034',linestyle=':',label='Held-out viseme limit 6%')
    ax.set_xticks(x,ids,rotation=40,ha='right');ax.set_ylabel('Lip landmark RMS / neutral mouth width (%)')
    ax.set_title('First mapping candidate fails the numerical screen on all 11 fits')
    ax.legend(fontsize=8);fig.savefig(out/'mapping-errors.png',dpi=160);plt.close(fig)
    table=[]
    for f in faces:
        low=next(v for v in sensitivity if v['face']==f['id'] and v['ridge']==.0001)
        h=f['heldout_speech']
        table.append(f"| {f['id']} | {max(f['calibration_lip_rms'].values())*100:.1f}% | "
                     f"{max(f['leave_one_viseme_out_lip_rms'].values())*100:.1f}% | "
                     f"{low['max_calibration_lip_rms']*100:.2f}% | "
                     f"{low['max_leave_one_out_lip_rms']*100:.1f}% | "
                     f"{h['max_correction_mouth_widths']*100:.1f}% |")
    pp_failed=[f['id'] for f in faces if not f['PP_valid_contact_example']]
    ms=[f['fit_ms'] for f in faces]
    report=f'''# ARKit to MediaPipe mapping feasibility results

Date: 2026-11-07. Status: **mapping not verified; do not change the v1 player path**.

The standalone probe used actual avatar viseme meshes, actual output from the pinned compact network, and the matching source rig. The first residual-linear mapping does not pass its numerical quality screen on any of the 11 tested fits. This rejects the current calibration recipe as ready for integration; it does not establish that a robust mapping is impossible.

## What was tested

- Eleven independently loaded fits: eight contrasting avatars plus the current and two historical fits of avatar `05a87620`. The set deliberately includes a naturally parted neutral mouth, stylized human faces, nonhuman faces, and an anime face. These are a stress set, not a random estimate of all avatar quality. Historical fits are used only as fit-variation tests, not asserted to be current published avatars.
- Seven calibration roles per fit: PP, aa, E, ih, oh, ou and FF. Cohort landmarks were used where present, with stable upper-face alignment into that fit's own reference. Original source-image landmarks remain separate.
- Source examples selected from 48 recorded Trevor/Pippa clips. Each example is an actual predicted vector, not an averaged or guessed ARKit vector. Selection is currently automatic by phone role plus measured source geometry; these assignments have not passed independent source-pose review.
- Forty-eight M3/F3 clips held out of mapping-example selection: {faces[0]['heldout_speech']['frames']:,} frames at 60 fps, about {faces[0]['heldout_speech']['frames']/60:.1f} seconds. Held out here refers to mapping calibration, not an assertion about all original neural-network training.
- Source reference coefficients measured from silent PCM. Its mapping to each target reference is exact. Only jaw, mouth and cheek-puff controls can affect destination speech geometry; eye and brow controls are excluded.
- Pinned ONNX SHA-256 `{d['protocol']['model_sha256']}`. A fresh full-clip replay matched cached coefficients exactly (maximum absolute difference {d['cache_replay_max_abs_error']}). The sandbox's newer active export was not silently substituted.

## Measurements

All landmark errors and correction magnitudes below are fractions of the target fit's neutral mouth width, expressed as percentages. The primary regularization parameter was fixed at 0.03. The 0.0001 arm is an explicitly exploratory sensitivity check, not a revised acceptance threshold.

| Fit | Primary calibration RMS max | Primary held-out viseme RMS max | Low-regularization calibration RMS max | Low-regularization held-out viseme RMS max | Primary maximum contact correction |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

The fixed limits are 3% calibration RMS, 6% held-out viseme RMS and 5% maximum contact correction. No fit passes the complete screen. The source example matrix has rank {faces[0]['svd']['rank']}; it cannot identify 52 independent expression controls. Unmeasured directions retain the source deformation prior.

![Calibration and held-out errors](mapping-errors.png)

Reducing regularization reproduces the selected calibration poses within 0.4% of mouth width, but held-out viseme errors remain large. Fitting the known poses therefore does not establish generalization. The selected aa and E source poses also have almost the same normalized aperture, while many target meshes express very different aa and E openings. This is direct evidence that a viseme label and one representative speech frame are insufficient to define a dependable coefficient-to-target contract.

## Lip contact and asset quality

The candidate contact constraint brings 99.7–100% of eligible active source-seal frames within the target gap limit. However, it can require 7–15% of a mouth width of movement on several fits, exceeding the predeclared 5% budget. Remaining penetration occurs on some other frames. Forcing lips to meet does not make the rest of the deformation correct, and is not accepted as evidence that the mapping works.

Contact eligibility excludes coefficient vectors within 0.15 of the source reference, measured over owned channels. Those reference-like frames have a separate rest-preservation requirement: a naturally parted target reference must not be forced shut. The held-out tracks contain {faces[0]['heldout_speech']['eligible_active_source_sealed_frames']:,} eligible active seal frames and {faces[0]['heldout_speech']['reference_like_source_sealed_frames']:,} reference-like source-seal frames. This is a geometric consistency check against the source rig, not independent lip-sync ground truth.

The existing PP meshes for {', '.join('`'+i+'`' for i in pp_failed)} exceed the chosen inner-lip gap or penetration tolerance. Some misses are marginal; the larger gaps need inspection. A PP label alone is not accepted as a closed-contact target. No original mesh was repaired or overwritten.

## Visual comparisons

Green contours show each actual stored viseme mesh; red contours show the primary mapping prediction at its assigned source coefficients. The background is the existing source viseme image. Cohort predictions are transformed back using stable face anchors before overlay. These are geometry comparisons, not altered textures or a new player renderer.

![Actual fitted visemes versus mapped contours](viseme-fit-comparison.png)

The source library poses below are rendered directly from the matching source rig. They remain candidate assignments requiring review, especially coarticulated vowels, rounded controls and the labiodental role.

![Observed source calibration poses](source-calibration-poses.png)

## Cost and limits

The residual solve and ownership mask alone took {min(ms):.3f}–{max(ms):.3f} ms per face in this Python/NumPy run on the current Mac. This excludes file reads, JSON decoding, alignment, source registration, model startup and per-frame constraints, and is not a browser or mobile startup benchmark. It supports treating the small solve as a modest cost, but quality remains unresolved.

The source registration uses a synthetic front render and fixed barycentric embedding, with nine explicitly paired source lip locations. Other landmark registration is approximate and needs independent review. Only image-plane fitting has been measured. Full mesh triangle safety, depth, smoothing, texture warping and the production blink/oral compositor have not passed a gate in this probe. No runtime integration is authorized by these results.

## Recommended next experiment

1. Review and improve source pose correspondence before any player refactor. Replace one automatically selected frame per phoneme role with a small geometry-conditioned library spanning independent aperture, spread, rounding, lip roll/press and jaw-with-lip-closure combinations. Retain actual model sequences as the final distribution check. Calibrate shared coefficient semantics on the source rig rather than choosing more regularization to hide poor examples.
2. Validate the canonical source mouth registration and its local frames with controlled coefficient sweeps. Separate uncertainty in transferred anatomy from uncertainty in coefficient assignment.
3. Select reliable actual contact/open/spread/rounding targets for each fit. Inspect the PP outliers and use another verified same-face closed pose where available; do not edit avatar masters during this experiment.
4. Compare a geometry-matched residual fit with a bounded local nonlinear residual/interpolation model if the linear fit still fails. Freeze the candidate and thresholds before evaluating independent clips/faces. Held-out target meshes must not be used to select their own test coefficient inputs.
5. Keep the player geometry refactor and ONNX integration blocked until a candidate passes reconstruction, interpolation, contact, correction-size and geometry-safety checks. Later, build the same verified mapping in the browser initially, then move it to avatar publishing as already agreed.

## Reproduction and preservation

Research branch: `research/mediapipe-mapping-feasibility` in the dedicated sandbox worktree. Scripts and run instructions are included in `mapping-feasibility-probe.zip`. Raw results, the fixed protocol, source pose definitions, input hashes and sensitivity results are included separately. The earlier exploratory v1/v2 runs exposed ownership/display and rest-eligibility issues; the report uses the corrected, explicitly versioned v3 protocol. Numerical quality limits were not relaxed.

Five invariant tests cover residual reconstruction, preservation of nonzero source reference and unobserved channels, contact direction/input preservation, coordinate round trips/stable alignment, and exact short/tail audio-window coverage. Cached ONNX predictions were checked against an exact replay. The original avatar meshes and pinned model checksums are rechecked after the run.

No player, TTS service, CDN publisher, source avatar mesh or deployment was changed. The primary player checkout already had unrelated location UI edits when inspected; this work does not modify or revert them.
'''
    (out/'mapping-feasibility-results.md').write_text(report)
    for name in ['mapping-errors.png','source-calibration-poses.png']:
        shutil.copy2(out/name,r/name)
    for name in ['viseme-fit-comparison.png','source-registration-overlay.png','results.json',
                 'protocol.json','inputs.json','inventory.json','source-library.json',
                 'regularization-sensitivity.json','registration.json']:
        shutil.copy2(r/name,out/name)
    # Save the human-readable report in the research checkout as well.
    docs=Path(__file__).resolve().parents[3]/'docs/plans'
    docs.mkdir(parents=True,exist_ok=True)
    local_report=report
    for name in ['mapping-errors.png','source-calibration-poses.png','viseme-fit-comparison.png']:
        local_report=local_report.replace(']('+name+')','](../../.local/'+r.name+'/'+name+')')
    (docs/'mediapipe-mapping-feasibility-results.md').write_text(local_report)
    print(out/'mapping-feasibility-results.md')


if __name__=='__main__':main()

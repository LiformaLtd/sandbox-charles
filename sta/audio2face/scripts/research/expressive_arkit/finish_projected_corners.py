"""Export and verify projected refinement, including the user's exact regression."""
import json
import torch
import onnx
from common import *
from guard_mouth_tails import GapGuardFace
from refine_mouth_tails import NaturalWaveform
from refine_projected_corners import DEST,CURRENT


def main():
    torch.set_num_threads(4)
    if (DEST/'model.onnx').exists():raise ValueError('Export already exists')
    net=GapGuardFace();net.load_state_dict(torch.load(DEST/'face.pt',map_location='cpu',weights_only=True));net.eval()
    model=NaturalWaveform(net).eval();style=torch.zeros(1,60,3);style[:,:,1]=1
    torch.onnx.export(model,(torch.zeros(1,16000),torch.zeros(1,60,10),style),str(DEST/'model.onnx'),input_names=['pcm','emotion','gaze_style'],output_names=['blendshapes','articulation'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(DEST/'model.onnx'))
    print('EXPORTED',sha(DEST/'model.onnx'),flush=True)
    import verify_mouth_tails as verify
    verify.DEST=DEST;verify.load_net=GapGuardFace
    try:verify.main()
    except ValueError:pass
    report=json.loads((DEST/'verification.json').read_text());old=json.loads((CURRENT/'verification.json').read_text());extra=[]
    for row in report['phrases']:
        previous=next(r for r in old['phrases'] if r['sample']==row['sample'] and r['emotion']==row['emotion'])
        row['previous_review_tail_error_mm']=previous['tail_corner_error_after_mm']
        row['previous_review_lip_rms_mm']=previous['new_lip_rms_mm']
        if row['new_lip_rms_mm']>previous['new_lip_rms_mm']*1.05:extra.append(row['reference_key']+' lip-gap RMS >5% versus current review')
        if row['emotion']=='neutral' and row['tail_corner_error_after_mm']>previous['tail_corner_error_after_mm']*1.05:extra.append(row['reference_key']+' lost neutral gain')
        if row['sample']=='appointment' and row['emotion']=='anger' and row['tail_corner_error_after_mm']>row['tail_corner_error_before_mm']+.001:extra.append('Strong-anger appointment still regresses versus pre-corner-refinement model')
    report['user_regression_checks_passed']=not extra
    report['user_regression_failures']=extra
    report['mouth_tail_failures']+=extra;report['violations']+=extra
    report['mouth_tail_checks_passed']=not report['mouth_tail_failures'];report['passed']=not report['violations']
    report['review_warnings']=[] if report['mouth_tail_checks_passed'] else report['mouth_tail_failures']
    (DEST/'verification.json').write_text(json.dumps(report,indent=2))
    print('USER CHECKS',report['mouth_tail_checks_passed'],report['mouth_tail_failures'],flush=True)
    if not report['mouth_tail_checks_passed']:raise ValueError('Projected refinement failed acceptance')

if __name__=='__main__':main()

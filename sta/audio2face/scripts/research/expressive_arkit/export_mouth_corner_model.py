"""Export isolated mouth refinement and run the same regression checks."""
from common import *
from mouth_corner_model import MouthCornerFace,DEST,CURRENT,TRAINED
import finish_projected_corners as finish

if __name__=='__main__':
    DEST.mkdir(exist_ok=True)
    import torch
    torch.save(MouthCornerFace().state_dict(),DEST/'face.pt')
    freeze(DEST/'protocol.json',dict(source_sha256=sha(__file__),model_source_sha256=sha(Path(__file__).with_name('mouth_corner_model.py')),
        current_sha256=sha(CURRENT/'model.onnx'),trained_sha256=sha(TRAINED/'model.onnx'),
        method='Freeze 25 non-mouth coefficients from previous review; project only 27 jaw/mouth coefficients to current lip gaps and [0,1] bounds',
        projection_iterations=256,regression='Appointment/anger was explicit development training; other four phrases were rechecked, not trained by this refinement'))
    finish.DEST=DEST;finish.GapGuardFace=MouthCornerFace;finish.main()

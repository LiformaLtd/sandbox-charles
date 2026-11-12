from common import *
import numpy as np
from data import TRAIN,DEV

def main():
 assert sha(BASELINE)==EXPECTED
 sources=[HERE/n for n in ['common.py','data.py','model.py','train.py','test_model.py','freeze_training.py','prepare_confirmation.py','run_confirmation.py']]
 sources += [APP/'scripts/research/data_scale'/n for n in ['features.py','mlx_backend.py','evaluate.py']]
 sources += [APP/'scripts/research/multivoice/closure_metrics.py',APP/'docs/plans/compact-recurrent-state.md',APP/'src/compact-mel.js',APP/'src/compact-conv.wgsl']
 arrays={}
 for name in TRAIN+DEV:
  root=SCALE/'features'/name
  for p in root.glob('*.npy'):arrays[str(p)]=sha(p)
  arrays[str(root/'complete.json')]=sha(root/'complete.json')
 freeze(OUT/'protocol.json',dict(baseline=str(BASELINE),baseline_sha256=EXPECTED,seed=37,steps=1240,schedule_sha256=sha(OUT/'schedule.npz'),sources={str(p.relative_to(APP)):sha(p) for p in sources},arrays=arrays,confirmation_audit_sha256=sha(OUT/'confirmation/audit.json')))
 for p in sources:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 print('Frozen training recipe, data, sources and fresh confirmation.',flush=True)
if __name__=='__main__':main()

from common import *
def main():
 assert sha(BASELINE)==EXPECTED
 import numpy as np
 from geometry import prepare
 np.testing.assert_array_equal(prepare()['active'],np.load(WORK/'arkit-student/rig-metrics.npz')['active'])
 check=json.loads((OUT/'surface-loss-check.json').read_text());assert check['loss_sha256']==sha(HERE/'loss.py') and check['cpu_mlx_gradient_relative_l2']<.001
 calibration=json.loads((OUT/'calibration.json').read_text());assert calibration['source_sha256']==sha(HERE/'calibrate.py')
 prior=json.loads((SCALE/'training-protocol.json').read_text())
 for name,digest in prior['array_sha256'].items():assert sha(SCALE/name)==digest,name
 arrays={};locks={}
 for name in ['old-train','new-train','common-dev']:
  root=OUT/'surface-maps'/name;meta=json.loads((root/'complete.json').read_text());assert sha(root/'stats.npy')==meta['array_sha256'];arrays[str((root/'stats.npy').relative_to(OUT))]=meta['array_sha256'];locks[name]=sha(root/'complete.json')
 sources=[HERE/n for n in ['common.py','geometry.py','teacher.py','label_surface.py','cache_surface.py','loss.py','check.py','calibrate.py','freeze_training.py','train.py','evaluate_experiment.py']]
 sources += [APP/'docs/plans/compact-lip-surface.md',APP/'scripts/research/data_scale/mlx_backend.py',APP/'scripts/research/data_scale/features.py']
 freeze(OUT/'training-protocol.json',dict(baseline=str(BASELINE),baseline_sha256=EXPECTED,allowed_seeds=[37],steps=3720,surface_coefficient=calibration['surface_coefficient'],calibration_sha256=sha(OUT/'calibration.json'),source_sha256={str(p.relative_to(APP)):sha(p) for p in sources},surface_array_sha256=arrays,surface_map_locks=locks,prior_protocol_sha256=sha(SCALE/'training-protocol.json'),surface_primary_lock_sha256=sha(OUT/'surface-primary-lock.json'),teacher_projection_parity_sha256=sha(OUT/'teacher-projection-parity.json'),fresh_audit_sha256=sha(OUT/'confirmation/audit.json')))
 for p in sources:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 print('Two arms, calibrated coefficient and all geometry inputs frozen.',flush=True)
if __name__=='__main__':main()

"""Lock this one sparse auxiliary-loss pilot before any student update."""
from common import *
def main():
 assert sha(BASELINE)==EXPECTED
 assert json.loads((OUT/'browser-gradient.json').read_text())['passes']
 assert json.loads((EXPERT/'confirmation/summary.json').read_text())['fixed-confirmation']['passes']
 calibration=json.loads((OUT/'calibration.json').read_text());assert calibration['bridge']['passes']
 freeze(OUT/'sources/model.py',(SCALE/'sources/model.py').read_bytes())
 sources=[HERE/'browser_renderer.mjs',HERE/'browser-render.html',APP/'src/rig.js']+[HERE/n for n in ['common.py','render.py','preflight.py','confirm_renderer.py','browser_auxiliary.py','browser_preflight.py','cache.py','calibrate.py','train.py','prepare_confirmation.py','run_confirmation.py','freeze_training.py','evaluate_experiment.py','inference.py']]+[APP/'scripts/research/data_scale'/n for n in ['features.py','mlx_backend.py','evaluate.py']]+[APP/'scripts/research/multivoice/closure_metrics.py',APP/'docs/plans/compact-syncnet-loss.md']
 arrays={}
 for name,digest in json.loads((SCALE/'training-protocol.json').read_text())['array_sha256'].items():
  assert sha(SCALE/name)==digest;arrays[name]=digest
 freeze(OUT/'training-protocol.json',dict(baseline=str(BASELINE),baseline_sha256=EXPECTED,allowed_seeds=[37],steps=3720,auxiliary_lambda=calibration['auxiliary_lambda'],auxiliary_frequency=4,gradient_estimator='16 Rademacher central differences epsilon.003; actual WebKit Three images; probe seed37000+step',auxiliary_frames=5,auxiliary_fps=25,ramp_steps=372,auxiliary_loss='relu(student_sync_distance - fitted_teacher_sync_distance)',alignment_ms=-40,auxiliary_voice_schedule='Rotate eight voices; first eligible member of its six-window main-batch group; no extra sampling RNG',selection='macro gap RMS + .25*(1-contactF1), shared development sets; epoch zero eligible',source_sha256={str(p.relative_to(APP)):sha(p) for p in sources},array_sha256=arrays,confirmation_audit_sha256=sha(OUT/'confirmation/audit.json'),preflight_sha256={n:sha(OUT/n) for n in ['renderer-compatibility.json','renderer-gradient.json','browser-gradient.json','calibration.json','cache-lock.json']}))
 for p in sources:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 print('Frozen unchanged architecture, same3720 updates and main batches; sparse training-only SyncNet.',flush=True)
if __name__=='__main__':main()

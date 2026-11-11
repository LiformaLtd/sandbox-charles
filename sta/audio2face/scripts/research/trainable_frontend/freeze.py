from common import *
from data import ROOTS
import platform,importlib.metadata as md

def main():
 if sha(BASELINE)!=EXPECTED:raise ValueError('Current model changed')
 sources={str(p.relative_to(APP)):sha(p) for p in HERE.glob('*.py')}
 for p in [APP/'scripts/research/data_scale/features.py',APP/'scripts/research/data_scale/mlx_backend.py',APP/'scripts/research/data_scale/evaluate.py',APP/'scripts/research/compact_ablation/model.py',APP/'scripts/research/compact_ablation/metrics.py',APP/'scripts/research/multivoice/closure_metrics.py']:sources[str(p.relative_to(APP))]=sha(p)
 data={};power={}
 for name in ROOTS:
  data[name]={p.name:sha(p) for p in (SCALE/'features'/name).iterdir() if p.is_file()}
  root=POWER/name;meta=json.loads((root/'complete.json').read_text())
  if sha(root/'power.npy')!=meta['power_sha256']:raise ValueError('Power cache changed')
  power[name]={p.name:sha(p) for p in root.iterdir() if p.is_file()}
 primary=Path('/Users/charleswiles/Source/liforma');protected=[primary/'sandboxes/sandbox-charles/sta/audio2face/.local/arkit-student/james-arkit-student-v4.onnx']+list((primary/'application/player.liforma.ai/src/lib/runtime/speech-shape').rglob('*'))
 protected=[p for p in protected if p.is_file() and '__pycache__' not in p.parts]
 write(OUT/'protocol.json',dict(issue='LIFORMA-105',baseline=str(BASELINE),baseline_sha256=EXPECTED,
  source_sha256=sources,data_sha256=data,power_sha256=power,power_root=str(POWER),protected_sha256={str(p):sha(p) for p in protected},
  environment=dict(platform=platform.platform(),python=sys.version,packages={n:md.version(n) for n in ['mlx','mlx-metal','numpy','torch','onnx','onnxruntime']}),
  seeds=[37,73,101],steps=3720,arms=['fixed','learned'],initialization='Identical current predictor and fixed mel bank; same bank weights in both arms',
  frontend='Direct bank weights shape64x257 initialized to current matrix. Learned weights projected nonnegative after each update, not clamped inside forward. Fixed arm stop-gradient; zero-decay frontend optimizer prevents drift. FFT/Hann/power/compression fixed.',
  optimizer='Predictor AdamW .00012→.000012,wd.015; frontend AdamW at .01 times predictor rate,wd0; separate norm1 clipping/states. No added losses.',
  selection='Minimum common-dev + replay-dev macro voice RMS + .25*(1-contactF1), epoch zero eligible',
  decision='Primary original-condition mean gap reduction>=3% vs paired fixed controls; no mean contact F1/opening/timed-closure-total loss; no engine mean gap regression; gain-condition mean gap no regression; historical mean gap within2% of current. Report all seeds; no promotion without new independent confirmation and browser/visual verification.',
  cohort_status='Existing held-out feature/scale/gain recordings, not fresh independent confirmation',
  heldout_cohorts={str(p):sha(p) for p in [DATA/'.local/compact-feature-v1/confirmation/variants/manifest.json',SCALE/'datasets/confirmation/manifest.json',DATA/'.local/compact-gain-v2/manifest.json']}))
 print('Frozen frontend experiment sources, data, transforms and protected files.',flush=True)
if __name__=='__main__':main()

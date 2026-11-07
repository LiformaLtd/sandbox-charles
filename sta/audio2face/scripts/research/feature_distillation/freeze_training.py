"""Freeze new recipe and reuse the exact audited data-scale inputs."""
from common import *
def main():
 assert sha(BASELINE)==EXPECTED
 prior=json.loads((SCALE/'training-protocol.json').read_text())
 for name,digest in prior['array_sha256'].items():
  if sha(SCALE/name)!=digest:raise ValueError('Prior feature array changed')
 check=json.loads((OUT/'backend-check.json').read_text());assert check['model_sha256']==sha(HERE/'model.py') and check['gradient_relative_l2']<.001
 sources=list(HERE.glob('*.py'))+[APP/'scripts/research/data_scale/mlx_backend.py',APP/'scripts/research/data_scale/features.py',APP/'docs/plans/compact-feature-distillation.md']
 hints={}
 for name in ['old-train','new-train']:
  root=OUT/'hints'/name;meta=json.loads((root/'complete.json').read_text());assert sha(root/'hints.npy')==meta['hints_sha256'];hints[name]=dict(meta_sha256=sha(root/'complete.json'),array_sha256=meta['hints_sha256'])
 freeze(OUT/'sources/model.py',(SCALE/'sources/model.py').read_bytes())
 protocol=dict(baseline=str(BASELINE),baseline_sha256=EXPECTED,source_sha256={str(p.relative_to(APP)):sha(p) for p in sources},hints=hints,prior_protocol_sha256=sha(SCALE/'training-protocol.json'),backend_check_sha256=sha(OUT/'backend-check.json'),allowed_seeds=[37],steps=3720,coefficient=.1,ramp_steps=372,confirmation_audit_sha256=sha(OUT/'confirmation/audit.json'),export_source_sha256=sha(OUT/'sources/model.py'))
 freeze(OUT/'training-protocol.json',protocol)
 for p in sources:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 print('Both arms frozen before optimization; source/cache hashes verified.',flush=True)
if __name__=='__main__':main()

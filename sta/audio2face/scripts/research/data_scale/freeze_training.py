"""Lock matched data/compute grid and all input caches before optimization."""
from pathlib import Path
import json
from prepare_corpus import APP,OUT,HERE,freeze,sha


def main():
    if not (OUT/'assembly-lock.json').exists() or not (OUT/'disjoint-audit.json').exists():raise ValueError('Missing corpus checks')
    check=json.loads((OUT/'mlx-backend-check.json').read_text())
    if check['source_sha256']!=sha(HERE/'mlx_backend.py') or check['gradient_relative_l2']>.001 or check['forward_max_delta']>2e-5:raise ValueError('Backend is not validated')
    source=[HERE/n for n in ['train.py','features.py','mlx_backend.py','check_backend.py','assemble.py','prepare_shard.py','prepare_corpus.py','prompts.py','audit_disjoint.py','freeze_training.py']]
    source += [APP/'docs/plans/compact-data-scale-grid.md']
    fingerprints={};arrays={}
    for name in ['old-train','new-train','common-dev','replay-train','extra-train','replay-dev']:
        root=OUT/'features'/name
        fingerprints[name]=sha(root/'complete.json')
        for p in root.glob('*.npy'):arrays[str(p.relative_to(OUT))]=sha(p)
    baseline=APP/'.local/compact-coverage-v2/coverage/runs/control-replay-s37/model.onnx'
    if sha(baseline)!='5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd':raise ValueError('Wrong baseline')
    protocol=dict(baseline=str(baseline),baseline_sha256=sha(baseline),source_sha256={str(p.relative_to(APP)):sha(p) for p in source},
        feature_fingerprints=fingerprints,array_sha256=arrays,assembly_lock_sha256=sha(OUT/'assembly-lock.json'),
        corpus_protocol_sha256=sha(OUT/'corpus/protocol.json'),backend_check_sha256=sha(OUT/'mlx-backend-check.json'),
        seed=37,allowed_seeds=[37,73,101],sampler_seed=1037,short_steps=744,long_steps=3720,evaluation_interval=124,backend='MLX 0.31.2 float32, MLX_ENABLE_TF32=0',
        budget_policy='Short is an exact prefix snapshot of long; common 3720-step cosine schedule')
    freeze(OUT/'training-protocol.json',protocol)
    for p in source:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
    print('Frozen four-arm grid before optimization; all six memory-mapped caches locked.')


if __name__=='__main__':main()

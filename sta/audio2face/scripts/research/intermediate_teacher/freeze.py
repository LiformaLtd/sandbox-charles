"""Freeze before training or reading held-out scores."""
from common import *
import platform
import numpy as np, onnx

def main():
    if sha(BASELINE)!=EXPECTED: raise ValueError('Unexpected current model')
    sources={str(p.relative_to(APP)):sha(p) for p in HERE.glob('*.py')}
    sources.update({str(p.relative_to(APP)):sha(p) for p in [APP/'scripts/research/data_scale/mlx_backend.py', APP/'scripts/research/data_scale/features.py',APP/'scripts/research/data_scale/evaluate.py',APP/'scripts/research/multivoice/closure_metrics.py',APP/'scripts/research/compact_ablation/model.py']})
    data={}
    for name in ['old-train','new-train','common-dev','replay-train','extra-train','replay-dev']:
        root=SCALE/'features'/name
        data[name]={p.name:sha(p) for p in root.iterdir() if p.is_file()}
    primary=Path('/Users/charleswiles/Source/liforma')
    protected=[primary/'sandboxes/sandbox-charles/sta/audio2face/.local/arkit-student/james-arkit-student-v4.onnx']
    protected+=list((primary/'application/player.liforma.ai/src/lib/runtime/speech-shape').rglob('*'))
    protected=[p for p in protected if p.is_file() and '__pycache__' not in p.parts]
    protocol=dict(issue='LIFORMA-104',baseline=str(BASELINE),baseline_sha256=EXPECTED,
      sources_note='Read-only current expanded-corpus caches; prior held-out recordings are regression cohorts, not fresh confirmation.',
      source_sha256=sources,data_sha256=data,protected_sha256={str(p):sha(p) for p in protected},
      platform=platform.platform(),numpy=np.__version__,onnx=onnx.__version__,
      mentor=dict(width=192,steps=7440,seed=37,initialization='Independent random extra channels; original 96-channel path preserved, no replication'),
      students=dict(width=96,steps=3720,seeds=[37,73,101],arms=['control','distill'],
        coefficient=.1,ramp_steps=372,feature_layers=[5,7],loss='Mean valid-frame cosine loss at two animation layers, training-only 96-to-192 heads'),
      optimizer=dict(name='AdamW',lr=.00012,end_lr=.000012,weight_decay=.015,clip=1),
      selection='Minimum macro voice gap RMS + .25*(1-contact F1) on common-dev plus replay-dev; epoch zero eligible',
      mentor_gate='At least 1% development gap RMS improvement versus baseline; macro F1 no worse than .005. Gate is a pilot screen, not proof of superiority.',
      decision='Distilled students must beat paired controls by at least 3% mean gap RMS; no contact-F1 loss; no worse original-condition opening RMS or well-timed closure total; no engine mean regression; no historical baseline RMS regression over 2%; all three seed effects reported. No promotion without fresh confirmation and visual review.',
      heldout_cohorts={str(p):sha(p) for p in [DATA/'.local/compact-feature-v1/confirmation/variants/manifest.json',SCALE/'datasets/confirmation/manifest.json',DATA/'.local/compact-gain-v2/manifest.json']})
    write(OUT/'protocol.json',protocol)
    print('Frozen source, data, protected artifacts and experimental protocol.',flush=True)
if __name__=='__main__':main()

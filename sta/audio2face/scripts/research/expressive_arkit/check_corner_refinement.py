"""Compare upper-face tempo and projection precision with the prior review model."""
import argparse
import json
import numpy as np
from pathlib import Path
from common import *
from infer_full import sequence
from diagnose_brows import temporal_metrics


def main(directory):
    parent=APP/'.local/mouth-tail-gap-guard-v1/model.onnx'
    with np.load(OUT/'calibration.npz') as z:shape=z['shape'];gap=z['gap_basis']
    regions=dict(brows=np.argsort(np.max(np.linalg.norm(shape[41:46],axis=2),axis=0))[-512:],
        lids=np.unique(np.concatenate([np.argsort(np.linalg.norm(shape[c],axis=1))[-256:] for c in [0,5,6,7,12,13]])))
    rows=[]
    for sample in json.loads((APP/'public/assets/expressive/review.json').read_text())['samples']:
        pcm=np.fromfile(APP/'public'/sample['pcm'].lstrip('/'),'<f4')
        for emotion in ['neutral','anger']:
            a=sequence(parent,pcm,emotion)['blendshapes'];b=sequence(directory/'model.onnx',pcm,emotion)['blendshapes']
            delta=np.einsum('fc,ck->fk',b-a,gap,optimize=False)*10
            row=dict(sample=sample['id'],emotion=emotion,max_gap_change_mm=float(np.abs(delta).max()),gap_change_rms_mm=float(np.sqrt(np.mean(delta**2))))
            for name,region in regions.items():
                old=temporal_metrics(np.einsum('fc,cvk->fvk',a,shape[:,region],optimize=False)*10)['fast_4_15_hz_rms']
                new=temporal_metrics(np.einsum('fc,cvk->fvk',b,shape[:,region],optimize=False)*10)['fast_4_15_hz_rms']
                row[name]=dict(before=old,after=new,ratio=new/old)
            rows.append(row)
    result=dict(model_sha256=sha(directory/'model.onnx'),parent_sha256=sha(parent),cases=rows,
        max_fast_motion_ratio={name:max(r[name]['ratio'] for r in rows) for name in regions},
        max_individual_gap_change_mm=max(r['max_gap_change_mm'] for r in rows))
    (directory/'upper-face-checks.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();main(args.directory)

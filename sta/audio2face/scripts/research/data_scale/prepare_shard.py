"""Apply balanced gain assignments before full-teacher labelling, without label reuse."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from prepare_corpus import OUT,freeze,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('shard');args=p.parse_args()
    base=OUT/'corpus/base-shards'/args.shard;root=OUT/'corpus/label-shards'/args.shard;root.mkdir(parents=True,exist_ok=True)
    parity=json.loads((base/'instrumentation-parity.json').read_text())
    if parity['max_abs_difference']!=0:raise ValueError('Teacher instrumentation changed output')
    for source in (base/'sources').rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts:freeze(root/'sources'/source.relative_to(base/'sources'),source.read_bytes())
    for name in ['assets.json','phone-classes.json','instrumentation-parity.json']:freeze(root/name,(base/name).read_bytes())
    if not (root/'assets').exists():(root/'assets').symlink_to(base/'assets',target_is_directory=True)
    rows=[]
    for source in json.loads((base/'manifest.json').read_text()):
        if sha(source['audio'])!=source['sha256_audio']:raise ValueError('Changed synthesis')
        wave=np.fromfile(source['audio'],'<f4');peak=float(np.abs(wave).max())
        conditions=[source['training_gain']] if source['split']=='train' else ['original','half','boost']+(['quarter'] if source['split']=='test' else [])
        for condition in conditions:
            requested={'original':1.,'half':.5,'boost':2.,'quarter':.25}[condition]
            gain=requested if requested<=1 or peak==0 else max(1.,min(requested,.98/peak))
            row=dict(source,stem=source['stem']+'-'+condition,condition=condition,requested_gain=requested,actual_gain=gain,original_audio_sha256=source['sha256_audio'])
            audio=root/'audio'/(row['stem']+'.f32');freeze(audio,(wave*np.float32(gain)).astype('<f4').tobytes())
            row.update(audio=str(audio),sha256_audio=sha(audio));rows.append(row)
    freeze(root/'manifest.json',rows)
    freeze(root/'source-lock.json',dict(code_sha256=sha(__file__),base_manifest_sha256=sha(base/'manifest.json'),manifest_sha256=sha(root/'manifest.json')))
    print(args.shard,len(rows),'gain-conditioned examples; every variant needs its own teacher track.')


if __name__=='__main__':main()

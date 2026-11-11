"""Add continuous aligned energy features to the frozen rest dataset."""
import json
import shutil
import numpy as np
import torch
from common import *
from speech_rest_energy_model import EnergyRestFace,EnergyRestWaveform,DEST


def main():
    DEST.mkdir(exist_ok=True);torch.set_num_threads(4)
    if (DEST/'energy-data-protocol.json').exists():raise ValueError('Energy data exists')
    source=APP/'.local/speech-rest-v2';wave=EnergyRestWaveform(EnergyRestFace()).eval()
    rows={r['stem']:r for r in json.loads((OUT/'manifest.json').read_text())};manifest=json.loads((APP/'public/assets/expressive/review.json').read_text());audio={};cache={}
    for split in ['train','dev','test','regression','quiet']:
        with np.load(source/(split+'.npz')) as z:d={k:z[k].copy() for k in z.files}
        energies=[];cursors={}
        if split=='quiet':
            rng=np.random.default_rng(601)
            for amplitude in [0.,1e-6,1e-5,3e-5,1e-4]:
                for _ in range(3):
                    pcm=rng.normal(0,amplitude,(1,16000)).astype(np.float32)
                    with torch.inference_mode():e=wave.energy(torch.from_numpy(pcm))[0].numpy()
                    energies.extend([e]*11)
        else:
            for case in d['cases']:
                stem=str(case).rsplit('--',1)[0];i=cursors.get(str(case),0);cursors[str(case)]=i+1;key=(stem,i)
                if key not in cache:
                    if stem not in audio:
                        path=APP/'public'/next(s['pcm'] for s in manifest['samples'] if s['id']=='appointment').lstrip('/') if stem=='appointment' else rows[stem]['audio']
                        audio[stem]=np.fromfile(path,'<f4')
                    pcm=audio[stem];start=(i-1)*8000;x=np.zeros((1,16000),np.float32);lo,hi=max(0,start),min(len(pcm),start+16000)
                    if hi>lo:x[0,lo-start:hi-start]=pcm[lo:hi]
                    with torch.inference_mode():cache[key]=wave.energy(torch.from_numpy(x))[0].numpy()
                energies.append(cache[key])
        d['energy']=np.stack(energies);np.savez_compressed(DEST/(split+'.npz'),**d)
        if (source/(split+'-base.npy')).exists():shutil.copyfile(source/(split+'-base.npy'),DEST/(split+'-base.npy'))
        print('energy',split,len(d['energy']),flush=True)
    shutil.copyfile(source/'geometry.npz',DEST/'geometry.npz')
    freeze(DEST/'energy-data-protocol.json',dict(source_sha256=sha(__file__),model_source_sha256=sha(Path(__file__).with_name('speech_rest_energy_model.py')),
        source_data_sha256={s:sha(source/(s+'.npz')) for s in ['train','dev','test','regression','quiet']},
        energy='Continuous log-power at20/100/200ms aligned to existing60fps grid; no hard threshold',quiet='Training targets recalibrated to frozen model mean geometry for quiet inputs'))

if __name__=='__main__':main()

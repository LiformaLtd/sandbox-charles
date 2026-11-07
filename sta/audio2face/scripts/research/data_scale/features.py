"""Low-memory waveform frontend caches; sampled minibatches copy from read-only maps."""
from pathlib import Path
import json,os,hashlib,importlib.util
import numpy as np
from prepare_corpus import APP,OUT,WORK,freeze,sha


def base_module():
    spec=importlib.util.spec_from_file_location('scale_frozen_model',OUT/'sources/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def build(name,root,split):
    dest=OUT/'features'/name;dest.mkdir(parents=True,exist_ok=True)
    rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']==split]
    fingerprint=hashlib.sha256((root/'manifest.json').read_bytes()+Path(__file__).read_bytes())
    for row in rows:
        audio_sha=sha(row['audio'])
        if row.get('sha256_audio') and row['sha256_audio']!=audio_sha:raise ValueError('Changed waveform')
        fingerprint.update(audio_sha.encode());fingerprint.update((root/'labels'/(row['stem']+'.npz')).read_bytes())
    digest=fingerprint.hexdigest();complete=dest/'complete.json'
    if complete.exists():
        if json.loads(complete.read_text())['input_sha256']!=digest:raise ValueError('Cache input changed')
        return
    base=base_module();active=np.load(WORK/'arkit-student/rig-metrics.npz')['active']
    sizes=[len(range(-8000,Path(r['audio']).stat().st_size//4-4000,8000)) for r in rows];n=sum(sizes)
    voices=sorted({r.get('engine','replay')+':'+r.get('voice','Easton') for r in rows})
    specs={'mel':((n,64,101),np.float32),'weights':((n,30,39),np.float32),'gaps':((n,30,9),np.float32),'valid':((n,30),np.bool_),'voice':((n,),np.int32)}
    arrays={k:np.lib.format.open_memmap(dest/(k+'.part.npy'),mode='w+',dtype=dtype,shape=shape) for k,(shape,dtype) in specs.items()}
    pending=[];indices=[];offsets=np.arange(15,45)*16000//60;cursor=0
    def flush():
        if not pending:return
        wave=np.stack(pending)
        windows=np.lib.stride_tricks.sliding_window_view(np.pad(wave,((0,0),(200,200))),400,axis=1)[:,::160]
        fft=np.fft.rfft(windows*base.HANN,n=512,axis=-1);power=(fft.real**2+fft.imag**2).astype(np.float32)
        mel=np.log1p(np.einsum('btf,mf->btm',power,base.MEL,optimize=False)*10).transpose(0,2,1)
        if not np.isfinite(mel).all():raise ValueError('Invalid frontend')
        arrays['mel'][indices]=mel;pending.clear();indices.clear()
    for row,count in zip(rows,sizes):
        pcm=np.fromfile(row['audio'],'<f4');position=0
        with np.load(root/'labels'/(row['stem']+'.npz')) as z:weights,gaps=z['weights'][:,active],z['gaps']
        if len(weights)!=(len(pcm)*60+15999)//16000 or gaps.shape!=(len(weights),9):raise ValueError('Invalid target length')
        for start in range(-8000,len(pcm)-4000,8000):
            wave=np.zeros(16000,np.float32);lo,hi=max(start,0),min(start+16000,len(pcm));wave[lo-start:hi-start]=pcm[lo:hi]
            valid=(start+offsets>=0)&(start+offsets<len(pcm));valid_count=int(valid.sum())
            arrays['valid'][cursor]=valid
            arrays['weights'][cursor]=0;arrays['gaps'][cursor]=0
            arrays['weights'][cursor,valid]=weights[position:position+valid_count];arrays['gaps'][cursor,valid]=gaps[position:position+valid_count]
            arrays['voice'][cursor]=voices.index(row.get('engine','replay')+':'+row.get('voice','Easton'))
            pending.append(wave);indices.append(cursor);cursor+=1;position+=valid_count
            if len(pending)==128:flush()
        if position!=len(weights):raise ValueError('Incomplete window crop')
    flush()
    for key in arrays:
        arrays[key].flush();(dest/(key+'.part.npy')).replace(dest/(key+'.npy'))
    freeze(complete,dict(input_sha256=digest,windows=n,voices=voices,rows=len(rows),source_sha256=sha(__file__)))
    print(name,n,'windows',len(rows),'recordings',flush=True)


class Dataset:
    def __init__(self,names):
        self.parts=[];self.voices=[];self.starts=[];self.length=0
        for name in names:
            root=OUT/'features'/name;meta=json.loads((root/'complete.json').read_text())
            data={k:np.load(root/(k+'.npy'),mmap_mode='r') for k in ['mel','weights','gaps','valid','voice']}
            self.starts.append(self.length);self.parts.append(data);self.length+=len(data['mel'])
            self.voices.extend(np.array(meta['voices'])[data['voice']].tolist())
        self.voices=np.array(self.voices)
    def batch(self,indices):
        # Advanced indexing produces writable copies; training never mutates maps.
        indices=np.asarray(indices);out={k:np.empty((len(indices),*self.parts[0][k].shape[1:]),self.parts[0][k].dtype) for k in ['mel','weights','gaps','valid']}
        for start,part in zip(self.starts,self.parts):
            mask=(indices>=start)&(indices<start+len(part['mel']))
            if mask.any():
                local=indices[mask]-start
                for key in out:out[key][mask]=part[key][local]
        return out


def main():
    source=APP/'.local/compact-coverage-v2/coverage/sources/model.py';freeze(OUT/'sources/model.py',source.read_bytes())
    for name,split in [('old-train','train'),('new-train','train'),('common-dev','dev')]:build(name,OUT/'datasets'/name,split)
    for name,root,split in [('replay-train',WORK/'arkit-student-v3','train'),('extra-train',WORK/'arkit-student-v3/extra','train'),('replay-dev',WORK/'arkit-student-v3','dev')]:build(name,root,split)


if __name__=='__main__':main()

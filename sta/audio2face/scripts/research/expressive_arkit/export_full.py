"""Export and verify the full-face model with its protected articulation stream."""
import sys
from pathlib import Path

APP=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(APP/'scripts/research/expressive_arkit'))
from common import *
from gaze import NEW,STYLES
from model import ExpressionAdapter
from gaze_model import GazeHead,FullFaceWaveform,FullFaceWithArticulation
from infer_full import read_audio,sequence
from fit import mm
from train import load
import numpy as np
import torch
import onnx
import onnxruntime as ort


def main():
    torch.set_num_threads(4)
    face=torch.load(OUT/'adapter.pt',map_location='cpu',weights_only=True)
    adapter=ExpressionAdapter(np.flatnonzero(face['speech_owned'].numpy()),np.zeros(96,np.float32),np.ones(96,np.float32))
    adapter.load_state_dict(face)
    gaze=GazeHead(face['feature_mean'],face['feature_scale'])
    gaze.load_state_dict(torch.load(NEW/'gaze.pt',map_location='cpu',weights_only=True))
    native=FullFaceWithArticulation(FullFaceWaveform(adapter,gaze)).eval()
    pcm=torch.zeros((1,16000));emotion=torch.zeros((1,60,10));style=torch.zeros((1,60,3));style[:,:,1]=1
    path=NEW/'fullface.onnx'
    torch.onnx.export(native,(pcm,emotion,style),str(path),input_names=['pcm','emotion','gaze_style'],
                      output_names=['blendshapes','articulation'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    old=ort.InferenceSession(str(BASELINE),sess_options=options,providers=['CPUExecutionProvider'])
    wav=Path('/Users/charleswiles/Source/liforma/sandboxes/sandbox-charles/sta/audio2face-2d/static/assets/speech/appointment-supertonic-m1.wav')
    audio=read_audio(wav)
    rng=np.random.default_rng(101)
    reports=[]
    for kind,pcm in [('silence',np.zeros((1,16000),np.float32)),('speech',audio[:16000][None,:]),
                     ('quiet-noise',rng.normal(0,.01,(1,16000)).astype(np.float32))]:
        baseline=old.run(None,{'pcm':pcm})[0]
        for identity in range(3):
            for ei in [-1,0,1,6]:
                emotion=np.zeros((1,60,10),np.float32)
                if ei>=0:emotion[:,:,ei]=1
                style=np.zeros((1,60,3),np.float32);style[:,:,identity]=1
                with torch.inference_mode():reference=native(torch.from_numpy(pcm),torch.from_numpy(emotion),torch.from_numpy(style))
                result=session.run(None,dict(pcm=pcm,emotion=emotion,gaze_style=style))
                parity=max(float(np.abs(v-t.numpy()).max()) for v,t in zip(result,reference))
                drift=float(np.abs(result[1]-baseline).max())
                if parity>3e-5 or drift>3e-5:raise ValueError('Full-face export/frozen speech parity failed')
                if any(not np.isfinite(v).all() or v.min()<0 or v.max()>1 for v in result):raise ValueError('Invalid full-face bounds')
                reports.append(dict(audio=kind,gaze_style=STYLES[identity],emotion_index=ei,
                                    native_onnx_max_error=parity,articulation_baseline_max_drift=drift))
    # Coverage of the trained outputs, including style-separated gaze.
    test=load('test');minima=np.ones(52,np.float32);maxima=np.zeros(52,np.float32)
    face_errors=[]
    for identity in range(3):
        for start in range(0,len(test['hidden']),128):
            stop=min(start+128,len(test['hidden']));h=torch.from_numpy(test['hidden'][start:stop]);b=torch.from_numpy(test['baseline'][start:stop])
            e=torch.from_numpy(test['emotion'][start:stop,None,:]).expand(-1,60,-1)
            s=torch.zeros(len(h),60,3);s[:,:,identity]=1
            with torch.inference_mode():
                w=adapter(h,b,e)*native.model.face_mask+gaze.coefficients(gaze(h,e,s))
            values=w[:,15:45].numpy()[test['valid'][start:stop]]
            minima=np.minimum(minima,values.min(0));maxima=np.maximum(maxima,values.max(0))
    names=json.loads((OUT/'channels.json').read_text())
    coverage=[dict(name=n,min=float(minima[i]),max=float(maxima[i]),range=float(maxima[i]-minima[i])) for i,n in enumerate(names)]
    # Exact original sentence; preserve the frozen track independently of style.
    regressions=[]
    base_tracks=np.load(OUT/'appointment-tracks.npz')['baseline']
    with np.load(OUT/'calibration.npz') as z:basis=z['gap_basis'];rest=z['gap_neutral']
    base_gap=mm(base_tracks,basis)+rest;base_contact=np.median(base_gap,axis=1)<=.1
    tracks={}
    for name in ['neutral','amazement','anger','joy']:
        result=sequence(path,audio,name,1.,'James')
        speech=result['articulation'];w=result['blendshapes']
        speech_gap=mm(speech,basis)+rest;full_gap=mm(w,basis)+rest
        speech_contact=np.median(speech_gap,axis=1)<=.1;full_contact=np.median(full_gap,axis=1)<=.1
        delta=float(np.abs(speech-base_tracks).max())
        if delta>3e-5 or np.any(speech_contact!=base_contact):raise ValueError('Articulation sentence regression failed')
        regressions.append(dict(emotion=name,articulation_max_drift=delta,
            articulation_changed_contact_frames=int((speech_contact!=base_contact).sum()),
            fullface_lost_contact_frames=int((base_contact&~full_contact).sum()),
            fullface_added_contact_frames=int((~base_contact&full_contact).sum())))
        tracks[name]=w
    np.savez_compressed(NEW/'appointment-tracks.npz',articulation=base_tracks,**tracks)
    before=json.loads((OUT/'protected-before.json').read_text())
    changed=[p for p,digest in before.items() if sha(p)!=digest]
    if changed:raise ValueError('Protected production files changed')
    protocol=json.loads((OUT/'training-protocol.json').read_text())
    face_source_changes=[name for name,digest in protocol['source_sha256'].items()
                         if sha(APP/'scripts/research/expressive_arkit'/name)!=digest]
    if face_source_changes:raise ValueError('Frozen face-training sources changed: '+str(face_source_changes))
    gaze_protocol=json.loads((NEW/'training-protocol.json').read_text())
    if any(sha(APP/'scripts/research/expressive_arkit'/n)!=d for n,d in gaze_protocol['source_sha256'].items()):
        raise ValueError('Frozen gaze-training sources changed')
    report=dict(model_sha256=sha(path),bytes=path.stat().st_size,outputs=['blendshapes [1,60,52]','articulation [1,60,52]'],
        inputs=['pcm [1,16000]','emotion [1,60,10]','gaze_style [1,60,3]'],
        native_onnx_cases=reports,trained_output_ranges=coverage,
        unexercised_outputs=[r['name'] for r in coverage if r['range']<.001],
        appointment_regression=regressions,protected_files_checked=len(before),protected_files_changed=changed,
        frozen_face_training_sources_unchanged=True,frozen_gaze_training_sources_unchanged=True,
        release_ready=False,head_rotation='deferred; no downloadable official NVIDIA head checkpoint found',
        limitations=['Strong emotion can still open the full expressive mouth through speech closures.',
          'The separate articulation output retains current speech timing/closures; a future avatar-specific contact compositor must use it before Player integration.',
          'Gaze and tongue calibrations are research estimates and need visual checking.',
          'Emotion and gaze-style inputs are explicit controls; no emotion classifier is trained here.',
          'No fresh independent confirmation, browser WebGPU validation, 2.5D upper-face mapping or Player rollout.'])
    freeze(NEW/'verification.json',report)
    print(json.dumps(dict(model_sha256=report['model_sha256'],bytes=report['bytes'],
        unexercised_outputs=report['unexercised_outputs'],max_parity=max(r['native_onnx_max_error'] for r in reports),
        protected_files_checked=len(before),appointment=regressions),indent=2))


if __name__=='__main__':main()

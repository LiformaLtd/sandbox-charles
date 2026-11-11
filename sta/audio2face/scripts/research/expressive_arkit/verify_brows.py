"""Prove mouth invariance and measure brow stability on all three James phrases."""
import copy
import json
import numpy as np
import torch
import onnx
from onnx import helper
import onnxruntime as ort
from common import *
from gaze import NEW
from train_brows import FIX,PARENT
from brow_model import BrowDecoder,BROW_IDS
from infer_full import sequence
from diagnose_brows import temporal_metrics


def parent_contract(parent,changed):
    if len(changed.graph.node)<=len(parent.graph.node):raise ValueError('Missing brow branch')
    for original,new in zip(parent.graph.node,changed.graph.node):
        restored=copy.deepcopy(new)
        for field in [restored.input,restored.output]:
            for i,name in enumerate(field):
                if name=='brow_stability/prior_blendshapes':field[i]='blendshapes'
        if original.SerializeToString()!=restored.SerializeToString():raise ValueError('Parent operation changed: '+original.name)
    original={v.name:v.SerializeToString() for v in parent.graph.initializer}
    updated={v.name:v.SerializeToString() for v in changed.graph.initializer}
    if any(updated.get(name)!=value for name,value in original.items()):raise ValueError('Parent initializer changed')
    return dict(operations_preserved=len(parent.graph.node),initializers_preserved=len(original))


def main():
    torch.set_num_threads(4)
    parent=onnx.load(PARENT);fixed=onnx.load(FIX/'model.onnx');onnx.checker.check_model(fixed)
    contract=parent_contract(parent,fixed)
    selected=json.loads((FIX/'selection.json').read_text());latent=selected['latent_value']
    parent.graph.output.append(helper.make_tensor_value_info(latent,onnx.TensorProto.FLOAT,[1,64,60]))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    debug=ort.InferenceSession(parent.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    old_session=ort.InferenceSession(str(PARENT),sess_options=options,providers=['CPUExecutionProvider'])
    new_session=ort.InferenceSession(str(FIX/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    checkpoint=torch.load(FIX/'brow-decoder.pt',map_location='cpu',weights_only=True)
    net=BrowDecoder(checkpoint['projection.weight'],checkpoint['projection.bias']);net.load_state_dict(checkpoint);net.eval()
    with np.load(OUT/'calibration.npz') as z:shape=z['shape'];neutral=z['neutral'];basis=z['gap_basis'];rest=z['gap_neutral']
    mouth=(np.abs(neutral[:,0])<4)&(neutral[:,1]>165)&(neutral[:,1]<171)&(neutral[:,2]>4)
    mouth_brow_influence=float(np.abs(shape[BROW_IDS][:,mouth]).max())
    if mouth_brow_influence>1e-7 or np.abs(basis[BROW_IDS]).max()!=0:raise ValueError('Brow shapes influence the mouth')
    support=np.max(np.linalg.norm(shape[BROW_IDS],axis=2),axis=0);region=np.argsort(support)[-512:]
    nonbrow=[i for i in range(52) if i not in BROW_IDS]
    manifest=json.loads((APP/'public/assets/expressive/review.json').read_text())
    cases=[];phrases=[];tracks={};teacher_neutral_improved=0
    for sample in manifest['samples']:
        path=APP/'public'/sample['pcm'].lstrip('/');pcm=np.fromfile(path,'<f4')
        if sha(path)!=sample['sha256']:raise ValueError('Changed speech recording')
        for ei in [-1,1,6,0]:
            for style in range(3):
                controls=np.zeros((1,60,10),np.float32)
                if ei>=0:controls[:,:,ei]=1
                gaze=np.zeros((1,60,3),np.float32);gaze[:,:,style]=1
                inputs=dict(pcm=pcm[:16000][None,:].copy(),emotion=controls,gaze_style=gaze)
                before=old_session.run(None,inputs);after=new_session.run(None,inputs)
                if not np.array_equal(before[0][:,:,nonbrow],after[0][:,:,nonbrow]):raise ValueError('Non-brow coefficients changed')
                if not np.array_equal(before[1],after[1]):raise ValueError('Articulation changed')
                hidden=debug.run([latent],inputs)[0]
                with torch.inference_mode():reference=net(torch.from_numpy(hidden),torch.from_numpy(controls)).numpy()
                error=float(np.abs(after[0][:,:,BROW_IDS]-reference).max())
                if error>3e-5:raise ValueError('Native brow/ONNX parity failed')
                cases.append(dict(sample=sample['id'],emotion_index=ei,style=style,native_onnx_max_error=error,nonbrow_max_change=0.,articulation_max_change=0.))
        cached=np.load(NEW/('brow-diagnostic-'+sample['id'])/'tracks.npz')
        for emotion in ['neutral','anger']:
            before=sequence(PARENT,pcm,emotion,1.,'Claire');after=sequence(FIX/'model.onnx',pcm,emotion,1.,'Claire')
            a,b=before['blendshapes'],after['blendshapes']
            if not np.array_equal(a[:,nonbrow],b[:,nonbrow]) or not np.array_equal(before['articulation'],after['articulation']):raise ValueError('Whole-phrase mouth/timing regression')
            s0=np.einsum('fc,cvk->fvk',a,shape[:,region],optimize=False)*10
            s1=np.einsum('fc,cvk->fvk',b,shape[:,region],optimize=False)*10
            m0,m1=temporal_metrics(s0),temporal_metrics(s1)
            teacher=cached['teacher_'+emotion+'_weights']
            truth=np.einsum('fc,cvk->fvk',teacher[:,BROW_IDS],shape[BROW_IDS][:,region],optimize=False)*10
            p0=np.einsum('fc,cvk->fvk',a[:,BROW_IDS],shape[BROW_IDS][:,region],optimize=False)*10
            p1=np.einsum('fc,cvk->fvk',b[:,BROW_IDS],shape[BROW_IDS][:,region],optimize=False)*10
            e0=float(np.sqrt(np.mean((p0-truth)**2)));e1=float(np.sqrt(np.mean((p1-truth)**2)))
            old_gaps=np.einsum('fc,ck->fk',a,basis,optimize=False)+rest
            new_gaps=np.einsum('fc,ck->fk',b,basis,optimize=False)+rest
            if not np.array_equal(old_gaps,new_gaps):raise ValueError('Lip aperture changed')
            if emotion=='anger' and m1['fast_4_15_hz_rms']>.3*m0['fast_4_15_hz_rms']:raise ValueError('Insufficient brow jitter reduction')
            if emotion=='neutral':
                if e1>=e0:raise ValueError('Neutral eyebrows are no closer to teacher')
                if m1['slow_0_4_hz_rms']<=1.2*m0['slow_0_4_hz_rms']:raise ValueError('Neutral brow motion not restored')
                teacher_neutral_improved+=1
            phrases.append(dict(sample=sample['id'],emotion=emotion,frames=len(a),before=m0,after=m1,
                teacher_brow_surface=temporal_metrics(truth),before_teacher_brow_rms_mm=e0,after_teacher_brow_rms_mm=e1,
                nonbrow_bit_identical=True,articulation_bit_identical=True,lip_aperture_bit_identical=True))
            tracks[sample['id']+'--'+emotion+'--before']=a;tracks[sample['id']+'--'+emotion+'--after']=b
    protected=json.loads((OUT/'protected-before.json').read_text());changed=[p for p,h in protected.items() if sha(p)!=h]
    if changed:raise ValueError('Protected primary files changed: '+str(changed))
    np.savez_compressed(FIX/'phrase-tracks.npz',**tracks)
    report=dict(parent_contract=contract,model_sha256=sha(FIX/'model.onnx'),bytes=(FIX/'model.onnx').stat().st_size,
        brow_ids=BROW_IDS,unchanged_output_count=47,mouth_brow_influence_model_units=mouth_brow_influence,
        native_onnx_cases=cases,phrases=phrases,neutral_teacher_improved_phrases=teacher_neutral_improved,
        protected_primary_files_checked=len(protected),protected_primary_files_changed=changed,
        limitations=['Only brows changed; other upper-face neutral gates remain as before.',
          'Known parent strong-emotion lip-contact and tongue-proxy issues are preserved, not repaired.',
          'Small existing corpus; no fresh independent confirmation; no Player or 2.5D rollout.'])
    freeze(FIX/'verification.json',report)
    print(json.dumps(dict(model_sha256=report['model_sha256'],bytes=report['bytes'],
        native_onnx_max_error=max(c['native_onnx_max_error'] for c in cases),phrases=phrases),indent=2))


if __name__=='__main__':main()

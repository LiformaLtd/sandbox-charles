"""Lock exports and prove baseline preservation and training/stream consistency."""
import json, os, sys
import numpy as np
import torch
from common import APP, OUT, BASELINE, sha, freeze
from data import backbone
from decoder import Decoder
from inference import infer
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper
import onnxruntime as ort


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        if sha(APP/name)!=digest:raise ValueError('Frozen source changed: '+name)
    reports={}; checkpoints={}; selection={}
    original={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer if v.name.startswith('student.')}
    rows=[r for r in json.loads((APP/'.local/compact-coverage-v2/coverage/manifest.json').read_text()) if r['split']=='dev']
    pcm=np.fromfile(rows[0]['audio'],'<f4')
    options=ort.SessionOptions();options.intra_op_num_threads=2
    baseline_session=ort.InferenceSession(str(BASELINE),sess_options=options,providers=['CPUExecutionProvider'])
    baseline=infer(baseline_session,pcm)
    torch.set_num_threads(2)
    student,_=backbone();norm=np.load(OUT/'normalization.npz')
    with np.load(OUT/'features/primary-dev.npz') as z:
        n=int(z['lengths'][0]); features=torch.from_numpy(z['features'][:n].copy())
    normalized=(features-torch.from_numpy(norm['center']))/torch.from_numpy(norm['spread'])
    padded=torch.cat((torch.zeros(7,96),normalized));contexts=torch.stack([padded[i:i+n] for i in range(8)],1)
    expected_base=((features@student.output.weight[...,0].T+student.output.bias)*student.scale+student.mean).clamp(0,1)
    for arm in ['control','temporal']:
        run=OUT/arm;report=json.loads((run/'training.json').read_text());checkpoints[arm]=torch.load(run/'checkpoint.pt',weights_only=False)
        if len(report['history'])!=24 or sha(run/'model.onnx')!=report['model_sha256']:raise ValueError('Incomplete/changed run')
        values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(run/'model.onnx').graph.initializer}
        for key,value in original.items():
            # Timeline buffers can be constant-folded; every learned tensor is checked.
            if key.endswith(('left','fraction','channel_map')):continue
            if key not in values or not np.array_equal(value,values[key]):raise ValueError('Backbone changed: '+key)
        decoder=Decoder(norm['center'],norm['spread'],arm);decoder.load_state_dict(torch.load(run/'decoder.pt',weights_only=False))
        with torch.no_grad():expected=(expected_base+decoder(contexts)*student.scale).clamp(0,1)@student.channel_map
        session=ort.InferenceSession(str(run/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        track=infer(session,pcm);delta=float(np.max(np.abs(track-expected.numpy())))
        if delta>2e-5:raise ValueError('Training/stream feature mismatch: '+str(delta))
        reports[arm]=dict(training_stream_max_delta=delta,change_from_v4=float(np.max(np.abs(track-baseline))))
        selection[arm]={k:report[k] for k in ['best_epoch','best_selection','model_sha256','bytes','parameters','decoder_parameters']}
    a,b=[checkpoints[n]['recipe'] for n in ['control','temporal']]
    if {k:v for k,v in a.items() if k!='arm'}!={k:v for k,v in b.items() if k!='arm'} or not torch.equal(checkpoints['control']['rng'],checkpoints['temporal']['rng']):raise ValueError('Unmatched training')
    if reports['control']['change_from_v4']>2e-5 and selection['control']['best_epoch']==0:raise ValueError('Zero correction changed baseline')
    freeze(OUT/'selection.json',selection)
    result=dict(selected=selection,checks=reports,frozen_backbone=True,identical_data_budget_sampler=True)
    (OUT/'training-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

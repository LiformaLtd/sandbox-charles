"""Evaluate frozen ONNX exports and rank training-set phoneme-pair errors."""
from pathlib import Path
import os
from collections import defaultdict
import importlib.util
import argparse
import json

import numpy as np
import onnxruntime as ort

APP=Path(__file__).resolve().parents[3]
WORK=Path(os.environ.get('A2F_RESEARCH_WORK',APP/'.local/research-corpora')).resolve()
import metrics
rig=np.load(WORK/'arkit-student/rig-metrics.npz')
basis,rest=rig['gap_basis'],rig['rest']+rig['gap_offset']

def session(path):
    options=ort.SessionOptions()
    options.intra_op_num_threads=2
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

def collect(root, model, cache, split):
    records=[]
    for row in json.loads((root/'manifest.json').read_text()):
        if row['split']!=split:continue
        with np.load(root/'labels'/f'{row["stem"]}.npz') as z:
            truth,weights=z['gaps'],z['weights']
        cached=cache/f'{row["stem"]}.npy'
        if cached.exists():
            pred=np.load(cached)
        else:
            pred=metrics.infer(model,np.fromfile(row['audio'],'<f4'))
            np.save(cached,pred)
        assert pred.shape==weights.shape
        active=rig['active']
        names=json.loads((WORK/'arkit-student/channels.json').read_text())
        lips=[i for i in active if names[i].startswith(('mouth','jaw'))]
        row['_coefficient_mae']=float(np.mean(np.abs(pred[:,active]-weights[:,active])))
        row['_mouth_mae']=float(np.mean(np.abs(pred[:,lips]-weights[:,lips])))
        records.append((row,truth,weights@basis+rest,pred@basis+rest))
    return records

def rank_pairs(records):
    pairs=defaultdict(lambda:dict(count=0,frames=0,squared=0.,contact_mismatch=0))
    for row,truth,fit,pred in records:
        times=np.arange(len(truth))/60
        for a,b in zip(row.get('phones',[]),row.get('phones',[])[1:]):
            if b['start']-a['end']>.12:continue
            mask=(times>=a['start'])&(times<b['end'])
            if not mask.any():continue
            p=pairs[a['phone']+' '+b['phone']]
            p['count']+=1
            p['frames']+=int(mask.sum())
            p['squared']+=float(np.sum((truth[mask]-pred[mask])**2))
            p['contact_mismatch']+=int(np.count_nonzero((np.median(truth[mask],axis=-1)<=.1)!=(np.median(pred[mask],axis=-1)<=.1)))
    result=[]
    for pair,v in pairs.items():
        result.append(dict(pair=pair,count=v['count'],
                           lip_gap_rmse=(v['squared']/(v['frames']*9))**.5,
                           contact_error=v['contact_mismatch']/v['frames']))
    return sorted(result,key=lambda x:x['lip_gap_rmse'],reverse=True)

def summarize(records):
    result={name:metrics.stats(records,index) for name,index in [('fit',2),('student',3)]}
    count=sum(len(r[1]) for r in records)
    for key in ('coefficient_mae','mouth_mae'):
        result['student'][key]=round(sum(r[0]['_'+key]*len(r[1]) for r in records)/count,5)
    return result

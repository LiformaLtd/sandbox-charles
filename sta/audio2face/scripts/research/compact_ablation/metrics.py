"""Compare identical browser exports against held-out teacher geometry."""
from pathlib import Path
import argparse
import json

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'arkit-student'
OFFSETS = np.arange(15,45)*16000//60
THRESHOLD = .10

def infer(session, audio):
    frames = []
    for start in range(-8000,len(audio)-4000,8000):
        valid = (start+OFFSETS>=0)&(start+OFFSETS<len(audio))
        if not valid.any():
            continue
        window = np.zeros((1,16000),np.float32)
        lo,hi = max(start,0),min(start+16000,len(audio))
        window[0,lo-start:hi-start] = audio[lo:hi]
        frames.append(session.run(None,{'pcm':window})[0][0,15:45][valid])
    return np.concatenate(frames)

def runs(closed):
    edges = np.flatnonzero(np.diff(np.r_[False,closed,False]))
    return list(zip(edges[::2],edges[1::2],strict=True))

def associate_events(phones, teacher_runs):
    """Use the aligner for phoneme identity and the teacher for exact closure frames."""
    pairs = []
    for i, event in enumerate(phones):
        if event['phone'] not in 'm b p'.split():
            continue
        for j, (a,b) in enumerate(teacher_runs):
            if (b-a) < 2 or b/60 < event['start']-.045 or a/60 > event['end']+.035:
                continue
            overlap = min(b/60,event['end']) - max(a/60,event['start'])
            distance = abs((a+b)/120-(event['start']+event['end'])/2)
            pairs.append((-overlap,distance,i,j))
    seen_phones,seen_runs = set(),set()
    for _,_,i,j in sorted(pairs):
        if i not in seen_phones and j not in seen_runs:
            seen_phones.add(i)
            seen_runs.add(j)
            yield phones[i]['phone'],teacher_runs[j]

def stats(rows, i):
    truth = np.concatenate([v[1] for v in rows])
    pred = np.concatenate([v[i] for v in rows])
    actual,closed = [np.median(g,axis=-1)<=THRESHOLD for g in (truth,pred)]
    hits = int((actual & closed).sum())
    return {
        'clips': len(rows),
        'frames': len(truth),
        'lip_gap_rmse': round(float(np.mean((truth-pred)**2)**.5),4),
        'contact_f1': round(2*hits/max(1,int(actual.sum()+closed.sum())),4),
        'contact_precision': round(hits/max(1,int(closed.sum())),4),
        'contact_recall': round(hits/max(1,int(actual.sum())),4),
        'lip_gap_velocity_mae': round(float(np.mean(np.concatenate([
            np.abs(np.diff(v[1],axis=0)-np.diff(v[i],axis=0)) for v in rows
        ]))),4),
    }

def event_scores(rows, i, align):
    result = {p: dict(events=0, hits=0, onset=[], release=[]) for p in ('m','b','p')}
    matched = {}
    for row, truth, fit, first, second in rows:
        values = [row,truth,fit,first,second]
        actual = runs(np.median(truth,axis=-1)<=THRESHOLD)
        predicted = runs(np.median(values[i],axis=-1)<=THRESHOLD)
        pairs = []
        for k, (phone,(a,b)) in enumerate(associate_events(align.get(row['stem'],[]),actual)):
            score = result[phone]
            score['events'] += 1
            for j, (c,d) in enumerate(predicted):
                overlap = min(b,d)-max(a,c)
                if overlap > 0:
                    pairs.append((-overlap,abs(a-c)+abs(b-d),k,j,phone,a,b,c,d))
        seen_events,seen_runs = set(),set()
        for _,_,k,j,phone,a,b,c,d in sorted(pairs):
            if k in seen_events or j in seen_runs:
                continue
            seen_events.add(k)
            seen_runs.add(j)
            matched[row['stem'],phone,a,b] = (abs(c-a),abs(d-b))
            result[phone]['hits'] += 1
            result[phone]['onset'].append(abs(c-a))
            result[phone]['release'].append(abs(d-b))
    for score in result.values():
        for edge in ('onset','release'):
            values = score.pop(edge)
            score[edge+'_mae_ms_on_hits'] = round(float(np.mean(values)*1000/60),1) if values else None
    return result, matched

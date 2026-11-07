"""One-shot gain follow-up evaluation, separate from development selection."""
from pathlib import Path
import argparse
from collections import defaultdict
import json
import os
import sys
import numpy as np
import onnxruntime as ort
from prepare import APP, OUT, frozen_write, sha

sys.path.insert(0, str(APP / 'scripts/research/compact_ablation'))
import metrics
sys.path.insert(0, str(APP / 'scripts/research/multivoice'))
import closure_metrics


def aggregate(rows):
    voices = sorted({r['engine'] + ':' + r['voice'] for r in rows})
    by_voice = {}
    for voice in voices:
        clips = [r for r in rows if r['engine'] + ':' + r['voice'] == voice]
        n = sum(r['gap_values'] for r in clips)
        a, b, hit = [sum(r[k] for r in clips) for k in ['teacher_contacts', 'student_contacts', 'contact_hits']]
        opening_n = sum(r['opening_values'] for r in clips)
        by_voice[voice] = dict(lip_gap_rmse_mm=10 * np.sqrt(sum(r['squared_gap_error'] for r in clips) / n),
                              contact_f1=2 * hit / max(1, a + b),
                              opening_rmse_mm=10 * np.sqrt(sum(r['opening_squared_error'] for r in clips) / max(1, opening_n)))
    return dict(macro_gap_rmse_mm=float(np.mean([v['lip_gap_rmse_mm'] for v in by_voice.values()])),
                macro_contact_f1=float(np.mean([v['contact_f1'] for v in by_voice.values()])),
                voices=by_voice, clips=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    frozen_write(OUT / 'sources/evaluate-gain.py', Path(__file__).read_bytes())
    if args.freeze_only:
        print('Evaluation source frozen before training.'); return
    work = Path(os.environ['A2F_RESEARCH_WORK'])
    protocol = json.loads((OUT / 'protocol.json').read_text())
    rig = np.load(work / 'arkit-student/rig-metrics.npz')
    basis, rest, active = rig['gap_basis'], rig['rest'] + rig['gap_offset'], rig['active']
    models = {'v4': Path(protocol['initialization']), **{
        arm: OUT / arm / 'runs/control-replay-s37/model.onnx' for arm in ['control', 'augmented']}}
    rows = [r for r in json.loads((OUT / 'manifest.json').read_text()) if r['split'] == 'test']
    results = {}
    for name, path in models.items():
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        cache = OUT / 'predictions' / sha(path); cache.mkdir(parents=True, exist_ok=True)
        details, records = [], defaultdict(list)
        for row in rows:
            if sha(row['audio']) != row['sha256_audio']:
                raise ValueError('Changed waveform')
            with np.load(OUT / 'labels' / (row['stem'] + '.npz')) as z:
                truth, weights = z['gaps'], z['weights']
                if str(z['audio_sha256']) != row['sha256_audio']:
                    raise ValueError('Teacher/audio mismatch')
            cached = cache / (row['sha256_audio'] + '.npy')
            if cached.exists():
                pred = np.load(cached)
            else:
                pred = metrics.infer(session, np.fromfile(row['audio'], '<f4')); np.save(cached, pred)
            if pred.shape != weights.shape or not np.isfinite(pred).all():
                raise ValueError('Invalid predictions')
            gaps = pred @ basis + rest
            a, b = np.median(truth, -1) <= .1, np.median(gaps, -1) <= .1
            opening = np.median(truth, -1) > .35
            detail = dict(stem=row['stem'], engine=row['engine'], voice=row['voice'], condition=row['condition'],
                          requested_gain=row['requested_gain'], actual_gain=row['actual_gain'], peak=row['peak'],
                          squared_gap_error=float(np.square(gaps - truth).sum()), gap_values=truth.size,
                          teacher_contacts=int(a.sum()), student_contacts=int(b.sum()), contact_hits=int((a & b).sum()),
                          coefficient_mae=float(np.abs(pred[:, active] - weights[:, active]).mean()),
                          opening_squared_error=float(np.square(gaps[opening] - truth[opening]).sum()),
                          opening_values=int(opening.sum()) * 9,
                          velocity_mae_mm=10 * float(np.abs(np.diff(gaps, axis=0) - np.diff(truth, axis=0)).mean()))
            details.append(detail)
            records[row['condition']].append((row, truth, gaps))
        conditions = {}
        for condition, record in records.items():
            clips = [r for r in details if r['condition'] == condition]
            conditions[condition] = dict(**aggregate(clips),
                engines={e: aggregate([r for r in clips if r['engine'] == e]) for e in ['phonon', 'supertonic']},
                closure_audit=closure_metrics.score(record, lead=.1)['summary'])
        result = dict(model_sha256=sha(path), bytes=path.stat().st_size,
                      protocol_sha256=sha(OUT / 'protocol.json'), evaluator_sha256=sha(__file__),
                      conditions=conditions, clips=details,
                      trained_gain_macro_rmse_mm=float(np.mean([conditions[g]['macro_gap_rmse_mm'] for g in ['half', 'boost']])))
        results[name] = result
        (OUT / ('evaluation-' + name + '.json')).write_text(json.dumps(result, indent=2))
        print(name, {c: round(v['macro_gap_rmse_mm'], 4) for c, v in conditions.items()}, flush=True)
    control, augmented, v4 = [results[n] for n in ['control', 'augmented', 'v4']]
    improvement = 1 - augmented['trained_gain_macro_rmse_mm'] / control['trained_gain_macro_rmse_mm']
    original_delta = augmented['conditions']['original']['macro_gap_rmse_mm'] / v4['conditions']['original']['macro_gap_rmse_mm'] - 1
    contact_delta = augmented['conditions']['original']['macro_contact_f1'] - v4['conditions']['original']['macro_contact_f1']
    engine_gains = {e: 1 - np.mean([augmented['conditions'][g]['engines'][e]['macro_gap_rmse_mm'] for g in ['half', 'boost']])
                   / np.mean([control['conditions'][g]['engines'][e]['macro_gap_rmse_mm'] for g in ['half', 'boost']])
                   for e in ['phonon', 'supertonic']}
    summary = dict(gain_improvement_vs_control=float(improvement), original_change_vs_v4=float(original_delta),
                   original_contact_f1_delta=float(contact_delta), engine_gain_improvements=engine_gains,
                   promising=bool(improvement >= .05 and original_delta <= .02 and contact_delta >= -.01 and min(engine_gains.values()) >= 0),
                   policy='One seed; previously inspected voices/texts; not independent release confirmation')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

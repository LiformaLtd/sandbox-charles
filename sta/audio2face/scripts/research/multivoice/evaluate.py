"""Frozen, speaker-disjoint pilot evaluation; no checkpoint selection here."""
from pathlib import Path
import argparse
from collections import defaultdict
import hashlib
import json
import os
import sys
import numpy as np
import onnxruntime as ort

APP = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('A2F_MULTIVOICE_OUTPUT', APP / '.local/compact-multivoice')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()
sys.path.insert(0, str(APP / 'scripts/research/compact_ablation'))
import metrics
import closure_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('model', type=Path)
    p.add_argument('--split', choices=['test', 'dev'], default='test')
    args = p.parse_args()
    model_sha = sha(args.model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
    rig = np.load(WORK / 'arkit-student/rig-metrics.npz')
    basis, rest, active = rig['gap_basis'], rig['rest'] + rig['gap_offset'], rig['active']
    rows = [r for r in json.loads((OUT / 'manifest.json').read_text()) if r['split'] == args.split]
    cache = OUT / 'predictions' / model_sha
    cache.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    details = []
    for row in rows:
        if sha(row['audio']) != row['sha256_audio']:
            raise ValueError('Audio changed')
        with np.load(OUT / 'labels' / (row['stem'] + '.npz')) as z:
            weights, truth = z['weights'], z['gaps']
        cached = cache / (row['sha256_audio'] + '.npy')
        if cached.exists():
            pred = np.load(cached)
        else:
            pred = metrics.infer(session, np.fromfile(row['audio'], '<f4'))
            np.save(cached, pred)
        if pred.shape != weights.shape or not np.isfinite(pred).all():
            raise ValueError('Invalid predicted track')
        fit, gaps = weights @ basis + rest, pred @ basis + rest
        record = (row, truth, fit, gaps, gaps)
        for key in ['all', row['engine'], row['gender'], row['engine'] + ':' + row['gender'],
                    'voice:' + row['engine'] + ':' + row['voice'], 'kind:' + row['kind']]:
            groups[key].append(record)
        a, b = np.median(truth, axis=-1) <= .1, np.median(gaps, axis=-1) <= .1
        details.append(dict(stem=row['stem'], text=row['text'], engine=row['engine'], voice=row['voice'],
                            gender=row['gender'], kind=row['kind'], frames=len(truth),
                            squared_gap_error=float(np.square(gaps - truth).sum()),
                            gap_values=truth.size, teacher_contacts=int(a.sum()), student_contacts=int(b.sum()),
                            contact_hits=int((a & b).sum()), coefficient_mae=float(np.abs(pred[:, active] - weights[:, active]).mean())))
    result = dict(name=args.name, model_sha256=model_sha, model_bytes=args.model.stat().st_size,
                  manifest_sha256=sha(OUT / 'manifest.json'), split=args.split,
                  labels_sha256={r['stem']: sha(OUT / 'labels' / (r['stem'] + '.npz')) for r in rows},
                  groups={k: dict(student=metrics.stats(v, 3), fitted_teacher=metrics.stats(v, 2)) for k, v in groups.items()},
                  clips=details, note='One training seed and only two unseen voices per engine; pilot, not broad generalization proof.')
    align = {r['stem']: r['phones'] for r in rows}
    result['events'], _ = metrics.event_scores(groups['all'], 3, align)
    result['legacy_event_policy'] = 'Historical 45ms/35ms association with greedy assignment; retained for audit, superseded by closure_timing_audit'
    closure_records = [(r[0], r[1], r[3]) for r in groups['all']]
    result['closure_timing_audit'] = closure_metrics.score(closure_records, lead=.100)
    result['closure_association_sensitivity'] = {str(lead): closure_metrics.score(closure_records, lead=lead)['summary']
                                                 for lead in [.045, .150]}
    result['macro_voice_gap_rmse'] = float(np.mean([v['student']['lip_gap_rmse'] for k, v in result['groups'].items() if k.startswith('voice:')]))
    result['macro_voice_contact_f1'] = float(np.mean([v['student']['contact_f1'] for k, v in result['groups'].items() if k.startswith('voice:')]))
    destination = OUT / 'evaluation'
    destination.mkdir(exist_ok=True)
    (destination / (args.name + '-' + args.split + '.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ['name', 'split', 'model_bytes', 'macro_voice_gap_rmse', 'macro_voice_contact_f1']}), flush=True)


if __name__ == '__main__':
    main()

"""Frozen new-transcript comparison plus separate historical regression checks."""
from collections import defaultdict
import argparse
import json
import sys
import numpy as np
import onnxruntime as ort
from common import APP, OUT, WORK, freeze, sha
from inference import infer
GAIN = APP / ".local/compact-gain-v2"
COVERAGE = APP / ".local/compact-coverage-v2"

sys.path.insert(0, str(APP / 'scripts/research/compact_ablation'))
import metrics
sys.path.insert(0, str(APP / 'scripts/research/multivoice'))
import closure_metrics


def aggregate(clips):
    voices = sorted({r['engine'] + ':' + r['voice'] for r in clips})
    by_voice = {}
    for voice in voices:
        rows = [r for r in clips if r['engine'] + ':' + r['voice'] == voice]
        n = sum(r['gap_values'] for r in rows)
        a, b, hits = [sum(r[k] for r in rows) for k in ['teacher_contacts', 'student_contacts', 'contact_hits']]
        opening_n = sum(r['opening_values'] for r in rows)
        by_voice[voice] = dict(gap_rmse_mm=10 * np.sqrt(sum(r['squared_gap_error'] for r in rows) / n),
                              contact_f1=2 * hits / max(1, a + b),
                              opening_rmse_mm=10 * np.sqrt(sum(r['opening_squared_error'] for r in rows) / max(1, opening_n)))
    return dict(macro_gap_rmse_mm=float(np.mean([v['gap_rmse_mm'] for v in by_voice.values()])),
                macro_contact_f1=float(np.mean([v['contact_f1'] for v in by_voice.values()])), voices=by_voice, clips=len(clips))


def evaluate_cohort(session, rows, label_root, cache, basis, rest, active):
    details, records = [], defaultdict(list)
    for row in rows:
        if sha(row['audio']) != row['sha256_audio']:
            raise ValueError('Changed waveform')
        with np.load(label_root / (row['stem'] + '.npz')) as z:
            truth, weights = z['gaps'], z['weights']
            if str(z['audio_sha256']) != row['sha256_audio']:
                raise ValueError('Teacher/audio mismatch')
        path = cache / (row['sha256_audio'] + '.npy')
        if path.exists():
            pred = np.load(path)
        else:
            pred = infer(session, np.fromfile(row['audio'], '<f4')); np.save(path, pred)
        if pred.shape != weights.shape or not np.isfinite(pred).all():
            raise ValueError('Invalid prediction')
        gaps = np.einsum('tc,ck->tk', pred, basis, optimize=False) + rest
        a, b = np.median(truth, -1) <= .1, np.median(gaps, -1) <= .1
        opening = np.median(truth, -1) > .35
        details.append(dict(stem=row['stem'], text=row['text'], engine=row['engine'], voice=row['voice'],
                            gender=row['gender'], kind=row['kind'], condition=row['condition'],
                            squared_gap_error=float(np.square(gaps - truth).sum()), gap_values=truth.size,
                            teacher_contacts=int(a.sum()), student_contacts=int(b.sum()), contact_hits=int((a & b).sum()),
                            opening_squared_error=float(np.square(gaps[opening] - truth[opening]).sum()),
                            opening_values=int(opening.sum()) * 9,
                            coefficient_mae=float(np.abs(pred[:, active] - weights[:, active]).mean()),
                            velocity_mae_mm=10 * float(np.abs(np.diff(gaps, axis=0) - np.diff(truth, axis=0)).mean())))
        records[row['condition']].append((row, truth, gaps))
    conditions = {}
    for condition, record in records.items():
        clips = [r for r in details if r['condition'] == condition]
        conditions[condition] = dict(**aggregate(clips),
            engines={e: aggregate([r for r in clips if r['engine'] == e]) for e in ['phonon', 'supertonic']},
            closure_audit=closure_metrics.score(record, lead=.1)['summary'])
    return dict(conditions=conditions, clips=details)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    freeze(OUT / 'sources/evaluate-temporal.py', __import__('pathlib').Path(__file__).read_bytes())
    if args.freeze_only:
        print('State evaluator frozen before confirmation scoring.'); return
    protocol = json.loads((OUT / 'protocol.json').read_text())
    models = {'v4': __import__('pathlib').Path(protocol['initialization']),
              'control': OUT / 'control/model.onnx',
              'temporal': OUT / 'temporal/model.onnx'}
    rig = np.load(WORK / 'arkit-student/rig-metrics.npz')
    basis, rest, active = rig['gap_basis'], rig['rest'] + rig['gap_offset'], rig['active']
    fresh = [r for r in json.loads((OUT / 'confirmation/variants/manifest.json').read_text()) if r['split'] == 'test']
    previous = [r for r in json.loads((GAIN / 'manifest.json').read_text()) if r['split'] == 'test']
    inspected = [r for r in json.loads((COVERAGE / 'variants/manifest.json').read_text()) if r['split'] == 'test']
    capacity_rows = [r for r in json.loads((APP / '.local/compact-capacity-v1/confirmation/variants/manifest.json').read_text()) if r['split'] == 'test']
    reports = {}
    for name, model in models.items():
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        cache = OUT / 'predictions' / sha(model); cache.mkdir(parents=True, exist_ok=True)
        report = dict(model_sha256=sha(model), bytes=model.stat().st_size, evaluator_sha256=sha(__file__),
                      capacity_regression=evaluate_cohort(session, capacity_rows, APP / '.local/compact-capacity-v1/confirmation/variants/labels', cache, basis, rest, active),
                      fresh=evaluate_cohort(session, fresh, OUT / 'confirmation/variants/labels', cache, basis, rest, active),
                      historical=evaluate_cohort(session, previous, GAIN / 'labels', cache, basis, rest, active),
                      coverage_regression=evaluate_cohort(session, inspected, COVERAGE / 'variants/labels', cache, basis, rest, active))
        reports[name] = report
        (OUT / ('evaluation-' + name + '.json')).write_text(json.dumps(report, indent=2))
        print(name, {c: round(s['macro_gap_rmse_mm'], 4) for c, s in report['fresh']['conditions'].items()}, flush=True)
    candidate, control, baseline = [reports[n] for n in ['temporal', 'control', 'v4']]
    score = lambda r, cohort, gain: r[cohort]['conditions'][gain]['macro_gap_rmse_mm']
    f1 = lambda r, cohort, gain: r[cohort]['conditions'][gain]['macro_contact_f1']
    improvement = 1 - score(candidate, 'fresh', 'original') / score(control, 'fresh', 'original')
    engine_gains = {e: 1 - candidate['fresh']['conditions']['original']['engines'][e]['macro_gap_rmse_mm']
                   / control['fresh']['conditions']['original']['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon', 'supertonic']}
    levels = lambda r: np.mean([score(r, 'fresh', g) for g in ['half', 'boost']])
    checks = dict(fresh_original_rms=improvement >= .05, no_engine_regression=min(engine_gains.values()) >= 0,
                  no_fresh_contact_regression=f1(candidate, 'fresh', 'original') >= f1(control, 'fresh', 'original'),
                  no_gain_regression=levels(candidate) <= levels(control),
                  historical_rms=score(candidate, 'historical', 'original') <= 1.02 * score(baseline, 'historical', 'original'),
                  historical_contact=f1(candidate, 'historical', 'original') >= f1(baseline, 'historical', 'original') - .01,
                  coverage_regression_rms=score(candidate, 'coverage_regression', 'original') <= 1.02 * score(baseline, 'coverage_regression', 'original'),
                  coverage_regression_contact=f1(candidate, 'coverage_regression', 'original') >= f1(baseline, 'coverage_regression', 'original') - .01)
    summary = dict(fresh_original_improvement_vs_control=float(improvement), fresh_engine_improvements=engine_gains,
                   fresh_gain_improvement_vs_control=float(1 - levels(candidate) / levels(control)),
                   historical_original_change_vs_v4=float(score(candidate, 'historical', 'original') / score(baseline, 'historical', 'original') - 1),
                   checks={k: bool(v) for k, v in checks.items()}, promising=bool(all(checks.values())),
                   interpretation='One frozen-backbone temporal-decoder seed; fresh confirmation transcripts/recordings; previously studied training-held-out voice presets')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

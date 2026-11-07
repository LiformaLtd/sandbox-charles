"""Paired crossed voice/text bootstrap and descriptive pilot comparison."""
from pathlib import Path
import json
import os
import numpy as np

APP = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('A2F_MULTIVOICE_OUTPUT', APP / '.local/compact-multivoice')).resolve()


def main():
    names = ['v3', 'control', 'phoneme', 'features']
    reports = {name: json.loads((OUT / 'evaluation' / (name + '-test.json')).read_text()) for name in names}
    text = sorted({c['text'] for c in reports['v3']['clips']})
    voices = sorted({c['engine'] + ':' + c['voice'] for c in reports['v3']['clips']})
    groups = [[i for i, v in enumerate(voices) if v.startswith(engine + ':')] for engine in ['phonon', 'supertonic']]
    arrays = {}
    for name, report in reports.items():
        lookup = {(c['engine'] + ':' + c['voice'], c['text']): c for c in report['clips']}
        arrays[name] = np.asarray([[[lookup[v, t][key] for key in
                                    ['squared_gap_error', 'gap_values', 'teacher_contacts', 'student_contacts', 'contact_hits']]
                                   for t in text] for v in voices])
    def score(a, vi, ti):
        sums = a[vi][:, ti].sum(axis=1)
        rmse = np.sqrt(sums[:, 0] / sums[:, 1]).mean()
        f1 = (2 * sums[:, 4] / np.maximum(1, sums[:, 2] + sums[:, 3])).mean()
        return np.array([rmse, f1])
    rng = np.random.default_rng(37)
    draws = []
    for _ in range(2000):
        vi = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        ti = rng.integers(len(text), size=len(text))
        draws.append((vi, ti))
    result = dict(summary={}, comparisons={}, bootstrap='2000 crossed voice/text draws; voices resampled within engine',
                  limitation='One training seed, two test voices per engine. Intervals do not establish population-wide or training-seed robustness.')
    for name in names:
        result['summary'][name] = dict(macro_voice_gap_rmse=reports[name]['macro_voice_gap_rmse'],
                                       macro_voice_contact_f1=reports[name]['macro_voice_contact_f1'],
                                       model_bytes=reports[name]['model_bytes'], events=reports[name]['events'])
        result['summary'][name]['closure_timing_audit'] = reports[name]['closure_timing_audit']['summary']
    for candidate, baseline in [('control', 'v3'), ('phoneme', 'v3'), ('features', 'v3'),
                                 ('phoneme', 'control'), ('features', 'control')]:
        values = np.asarray([score(arrays[candidate], vi, ti) - score(arrays[baseline], vi, ti) for vi, ti in draws])
        result['comparisons'][candidate + '-' + baseline] = dict(
            gap_rmse_delta_95ci=np.quantile(values[:, 0], [.025, .975]).tolist(),
            contact_f1_delta_95ci=np.quantile(values[:, 1], [.025, .975]).tolist())
    dev = {}
    for arm in names[1:]:
        report = json.loads((OUT / 'runs' / (arm + '-s37') / 'training.json').read_text())
        dev[arm] = report['best_selection']
    result['primary_by_validation'] = min(dev, key=dev.get)
    result['validation_selection'] = dev
    result['followups'] = {}
    for name, run in [('easton-control', 'easton-control-s37'), ('replay', 'control-replay-s37')]:
        report = json.loads((OUT / 'evaluation' / (name + '-test.json')).read_text())
        training = json.loads((OUT / 'runs' / run / 'training.json').read_text())
        result['followups'][name] = dict(macro_voice_gap_rmse=report['macro_voice_gap_rmse'],
                                         macro_voice_contact_f1=report['macro_voice_contact_f1'],
                                         best_epoch=training['best_epoch'],
                                         closure_timing_audit=report['closure_timing_audit']['summary'],
                                         status='Post-pilot engineering diagnostic, not independent confirmation')
    (OUT / 'pilot-summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

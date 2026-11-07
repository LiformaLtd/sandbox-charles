"""Check transformation, label provenance and original-target numerical parity."""
from collections import defaultdict
import json
import numpy as np
from prepare import OUT, PILOT, sha


def main():
    rows = json.loads((OUT / 'manifest.json').read_text())
    recipes, groups, original_labels = set(), defaultdict(list), {}
    old_delta = dict(weights=0., gaps=0.)
    for row in rows:
        audio = np.fromfile(row['audio'], '<f4')
        if sha(row['audio']) != row['sha256_audio'] or not np.isfinite(audio).all():
            raise ValueError('Changed or nonfinite audio')
        if np.max(np.abs(audio)) > 1.000001:
            raise ValueError('Transformation exceeded full scale')
        with np.load(OUT / 'labels' / (row['stem'] + '.npz')) as z:
            if str(z['audio_sha256']) != row['sha256_audio']:
                raise ValueError('Label/audio mismatch')
            weights, gaps = z['weights'], z['gaps']
            recipes.add(str(z['recipe_sha256']))
            if not np.isfinite(weights).all() or not np.isfinite(gaps).all():
                raise ValueError('Nonfinite teacher labels')
        if row['condition'] == 'original':
            with np.load(PILOT / 'labels' / (row['original_stem'] + '.npz')) as previous:
                for key, value in [('weights', weights), ('gaps', gaps)]:
                    old_delta[key] = max(old_delta[key], float(np.abs(value - previous[key]).max()))
            original_labels[row['original_stem']] = gaps
        else:
            reference = original_labels[row['original_stem']]
            groups[row['condition']].append(dict(squared_delta=float(np.square(gaps - reference).sum()), values=gaps.size,
                                                  requested=row['requested_gain'], actual=row['actual_gain']))
    if len(recipes) != 1:
        raise ValueError('Mixed teacher-label recipes')
    result = dict(recordings=len(rows), label_recipe_sha256=next(iter(recipes)), original_label_max_abs_delta=old_delta,
                  teacher_gain_delta_mm={g: 10 * np.sqrt(sum(x['squared_delta'] for x in v) / sum(x['values'] for x in v)) for g, v in groups.items()},
                  amplification_limited_by_headroom=sum(r['condition'] == 'boost' and r['actual_gain'] < 2 for r in rows),
                  max_waveform_peak=max(r['peak'] for r in rows))
    logs = [OUT / arm / 'runs/control-replay-s37/training.json' for arm in ['control', 'augmented']]
    if all(p.exists() for p in logs):
        control, augmented = [json.loads(p.read_text()) for p in logs]
        if control['recipe']['config'] != augmented['recipe']['config'] or control['initial_dev'] != augmented['initial_dev']:
            raise ValueError('Unmatched configuration or initial development scores')
        for report in [control, augmented]:
            if report['steps_per_epoch'] != 31 or len(report['history']) != 24:
                raise ValueError('Unmatched completed update budget')
        paths = [OUT / arm / ('features-dev-' + report['recipe']['dev_sha256'][:16] + '.npz')
                 for arm, report in zip(['control', 'augmented'], [control, augmented])]
        with np.load(paths[0]) as a, np.load(paths[1]) as b:
            if a.files != b.files:
                raise ValueError('Development schemas differ')
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        result['matched_training_audit'] = dict(development_arrays_identical=True, initial_development_scores_identical=True,
                                                configuration_identical=True, updates_per_arm=744)
    (OUT / 'corpus-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

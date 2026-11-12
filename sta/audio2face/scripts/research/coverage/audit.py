"""Validate actual arm inputs, split isolation and common teacher provenance."""
from collections import Counter
import json
import numpy as np
from prepare import OUT, normalize, sha


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed: ' + path)
    manifests = {a: json.loads((OUT / a / 'manifest.json').read_text()) for a in ['control', 'coverage']}
    dev = [[r for r in rows if r['split'] == 'dev'] for rows in manifests.values()]
    if dev[0] != dev[1]:
        raise ValueError('Development cohorts differ')
    recipes, arms = set(), {}
    for arm, rows in manifests.items():
        split_texts = {s: {normalize(r['text']) for r in rows if r['split'] == s} for s in ['train', 'dev', 'test']}
        if any(split_texts[a] & split_texts[b] for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]):
            raise ValueError('Transcript leakage')
        counts = Counter(r['engine'] + ':' + r['voice'] for r in rows if r['split'] == 'train')
        if len(counts) != 8 or set(counts.values()) != {96}:
            raise ValueError('Unbalanced training voices')
        for row in rows:
            if sha(row['audio']) != row['sha256_audio']:
                raise ValueError('Changed audio')
            with np.load(OUT / arm / 'labels' / (row['stem'] + '.npz')) as z:
                if str(z['audio_sha256']) != row['sha256_audio'] or not np.isfinite(z['weights']).all() or not np.isfinite(z['gaps']).all():
                    raise ValueError('Teacher/audio mismatch or nonfinite targets')
                recipes.add(str(z['recipe_sha256']))
        arms[arm] = dict(train_recordings=sum(r['split'] == 'train' for r in rows),
                         train_distinct_texts=len(split_texts['train']), voices=dict(counts),
                         dev_recordings=len(dev[0]), test_recordings=sum(r['split'] == 'test' for r in rows))
    if len(recipes) != 1:
        raise ValueError('Mixed teacher recipes')
    report = dict(arms=arms, shared_development_identical=True, label_recipe_sha256=next(iter(recipes)),
                  quarter_gain_excluded_from_training=all(r['condition'] != 'quarter' for rows in manifests.values() for r in rows if r['split'] == 'train'))
    logs = [OUT / a / 'runs/control-replay-s37/training.json' for a in manifests]
    if all(p.exists() for p in logs):
        a, b = [json.loads(p.read_text()) for p in logs]
        if a['recipe']['config'] != b['recipe']['config'] or a['initial_dev'] != b['initial_dev']:
            raise ValueError('Unmatched training configuration/initial development')
        for log in [a, b]:
            if len(log['history']) != 24 or log['steps_per_epoch'] != 31:
                raise ValueError('Unmatched update budget')
        paths = [OUT / arm / ('features-dev-' + log['recipe']['dev_sha256'][:16] + '.npz') for arm, log in zip(manifests, [a, b])]
        with np.load(paths[0]) as x, np.load(paths[1]) as y:
            if x.files != y.files:
                raise ValueError('Development tensor schemas differ')
            for key in x.files:
                np.testing.assert_array_equal(x[key], y[key])
        report['completed_training_audit'] = dict(configuration_identical=True, dev_arrays_identical=True, updates_per_arm=744)
    (OUT / 'input-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

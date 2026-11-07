"""Post-hoc error decomposition; never a checkpoint-selection criterion."""
import json
from collections import defaultdict
import numpy as np
from prepare import OUT, WORK, sha


def main():
    rig = np.load(WORK / 'arkit-student/rig-metrics.npz')
    basis, rest = rig['gap_basis'], rig['rest'] + rig['gap_offset']
    rows = [r for r in json.loads((OUT / 'confirmation/variants/manifest.json').read_text()) if r['condition'] == 'original']
    result = dict(method='Post-hoc; same nine vertical gaps, macro voice; fitting is a reference, not a strict error floor', voices={})
    totals = defaultdict(lambda: np.zeros(5, np.float64))
    digest = json.loads((OUT / 'selection.json').read_text())['state']['model_sha256']
    for row in rows:
        if sha(row['audio']) != row['sha256_audio']:
            raise ValueError('Changed audio')
        with np.load(OUT / 'confirmation/variants/labels' / (row['stem'] + '.npz')) as z:
            raw, weights = z['gaps'], z['weights']
        fit = np.einsum('tc,ck->tk', weights, basis, optimize=False) + rest
        pred = np.load(OUT / 'predictions' / digest / (row['sha256_audio'] + '.npy'))
        gaps = np.einsum('tc,ck->tk', pred, basis, optimize=False) + rest
        a, b = fit - raw, gaps - fit
        totals[row['engine'] + ':' + row['voice']] += [np.square(a).sum(), np.square(b).sum(), 2 * (a * b).sum(), np.square(gaps - raw).sum(), raw.size]
    for voice, (fit, student, cross, total, count) in totals.items():
        if not np.isclose(fit + student + cross, total, rtol=1e-5):
            raise ValueError('Decomposition failed')
        result['voices'][voice] = dict(fitting_rms_mm=float(10 * np.sqrt(fit / count)), student_to_fit_rms_mm=float(10 * np.sqrt(student / count)), student_to_raw_rms_mm=float(10 * np.sqrt(total / count)), cross_mse_mm2=float(100 * cross / count))
    result['macro'] = {k: float(np.mean([v[k] for v in result['voices'].values()])) for k in next(iter(result['voices'].values()))}
    (OUT / 'fitting-diagnostic.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

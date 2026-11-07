"""Render the six protocol-defined fresh cases with one audio/frame clock."""
from pathlib import Path
import json
import os
import wave
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import closure_metrics

HERE = Path(__file__).resolve().parent
APP = HERE.parents[2]
CONFIRM = Path(os.environ.get('A2F_CONFIRMATION_ROOT', APP / '.local/compact-v4-confirmation')).resolve()
OUT = Path(os.environ.get('A2F_VALIDATION_OUTPUT', APP / '.local/compact-v4-validation')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()


def main():
    protocol = json.loads((CONFIRM / 'protocol.json').read_text())
    selected = json.loads((OUT / 'selection.json').read_text())['representative']
    baseline = json.loads((CONFIRM / 'evaluation/v3-test.json').read_text())
    models = [dict(id='teacher', title='Full teacher fitted to James'),
              dict(id='v3', title='Released compact v3', sha256=baseline['model_sha256']),
              dict(id='v4', title=f'Compact v4 candidate · seed {selected["seed"]}', sha256=selected['model_sha256'])]
    target = OUT / 'browser/web/public/review'
    target.mkdir(parents=True, exist_ok=True)
    rows = {r['stem']: r for r in json.loads((CONFIRM / 'manifest.json').read_text())}
    rig = np.load(WORK / 'arkit-student/rig-metrics.npz')
    basis, rest = rig['gap_basis'], rig['rest'] + rig['gap_offset']
    cases = []
    for stem in protocol['visual_cases']:
        row = rows[stem]
        with np.load(CONFIRM / 'labels' / (stem + '.npz')) as z:
            truth, teacher = z['gaps'], z['weights']
        tracks = {'teacher': teacher}
        for model in models[1:]:
            tracks[model['id']] = np.load(CONFIRM / 'predictions' / model['sha256'] / (row['sha256_audio'] + '.npy'))
        for model in models:
            values = tracks[model['id']]
            if values.shape != teacher.shape or not np.isfinite(values).all():
                raise ValueError('Invalid comparison track')
            values.astype('<f4').tofile(target / (stem + '-' + model['id'] + '.f32'))
        pcm = np.fromfile(row['audio'], '<f4')
        with wave.open(str(target / (stem + '.wav')), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())
        associations = closure_metrics.associate(row['phones'], closure_metrics.runs(np.median(truth, axis=-1) <= .1), len(truth))
        wanted = row['kind'].removeprefix('closure-') if row['kind'].startswith('closure-') else None
        events = [a for a in associations if a['status'] == 'unique' and (wanted is None or a['phone']['phone'] == wanted)]
        if wanted and events:
            event = events[0]
            a, b = event['event']; phone = event['phone']['phone']
            centre = (a + b - 1) // 2
            snapshots = [dict(frame=max(0, a - 4), title=f'Before /{phone}/'),
                         dict(frame=centre, title=f'Teacher contact /{phone}/'),
                         dict(frame=min(len(truth) - 1, b + 4), title=f'After /{phone}/')]
        else:
            centre = int(np.median(truth, axis=-1).argmax())
            snapshots = [dict(frame=max(0, centre - 4), title='Before teacher peak'),
                         dict(frame=centre, title='Maximum teacher opening'),
                         dict(frame=min(len(truth) - 1, centre + 4), title='After teacher peak')]
        fig, ax = plt.subplots(figsize=(12, 2.8), layout='constrained')
        times = np.arange(len(truth)) / 60
        ax.plot(times, np.median(truth, axis=-1) * 10, color='black', linewidth=1.8, label='Raw teacher')
        for model, color in zip(models, ['#999999', '#9a89b5', '#2b8cbe']):
            gaps = np.einsum('fc,cg->fg', tracks[model['id']], basis, optimize=False) + rest
            ax.plot(times, np.median(gaps, axis=-1) * 10, color=color, linewidth=1.1, label=model['title'])
        ax.axhline(1., color='#b2182b', linestyle='--', linewidth=.8)
        ax.axvline(centre / 60, color='#777777', linestyle=':', linewidth=.8)
        ax.set(xlabel='Time (s)', ylabel='Median vertical lip gap (mm)', xlim=(0, len(pcm) / 16000))
        ax.legend(ncol=4, fontsize=8)
        fig.savefig(target / (stem + '-gaps.png'), dpi=160); plt.close(fig)
        cases.append(dict(stem=stem, title=row['engine'].title() + ' ' + row['voice'] + ' · ' + row['kind'],
                          text=row['text'], frames=len(teacher), snapshots=snapshots))
    (target / 'manifest.json').write_text(json.dumps(dict(models=models, cases=cases, provisional=False), indent=2))
    page = OUT / 'browser/web/review.html'
    source = page.read_text().replace('repeat(5,minmax(0,1fr))', 'repeat(3,minmax(0,1fr))')
    page.write_text(source.replace('Matched mouth comparison', 'Fresh confirmation · v3 and v4 candidate'))
    print('Built six frozen fresh comparisons, representative selected by development scores.')


if __name__ == '__main__':
    main()

"""Six predetermined multi-voice clips, six synchronized James render lanes."""
from pathlib import Path
import hashlib
import json
import os
import sys
import wave
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

APP = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('A2F_MULTIVOICE_OUTPUT', APP / '.local/compact-multivoice')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()
sys.path.insert(0, str(APP / 'scripts/research/compact_ablation'))
import metrics

STEMS = ['test-phonon-hugh-000', 'test-phonon-freya-002', 'test-supertonic-m5-004',
         'test-supertonic-f5-007', 'test-phonon-freya-009', 'test-supertonic-f5-010']


def main():
    target = OUT / 'browser/web/public/review'
    target.mkdir(parents=True, exist_ok=True)
    models = [dict(id='teacher', title='Full teacher fitted to James'),
              dict(id='v3', title='Released compact v3', path=WORK / 'arkit-student-v3/james-arkit-student-v3.onnx')]
    for arm in ['control', 'phoneme', 'features']:
        models.append(dict(id=arm, title={'control': 'Multi-voice data only', 'phoneme': '+ Phoneme supervision',
                                         'features': '+ Speech-feature distillation'}[arm],
                           path=OUT / 'runs' / (arm + '-s37') / 'model.onnx'))
    models.append(dict(id='replay', title='Multi-voice + clean Easton replay',
                       path=OUT / 'runs/control-replay-s37/model.onnx'))
    for model in models[1:]:
        model['sha256'] = hashlib.sha256(model.pop('path').read_bytes()).hexdigest()
    rows = {r['stem']: r for r in json.loads((OUT / 'manifest.json').read_text())}
    rig = np.load(WORK / 'arkit-student/rig-metrics.npz')
    basis, rest = rig['gap_basis'], rig['rest'] + rig['gap_offset']
    cases = []
    for stem in STEMS:
        row = rows[stem]
        with np.load(OUT / 'labels' / (stem + '.npz')) as z:
            truth, teacher = z['gaps'], z['weights']
        tracks = {'teacher': teacher}
        for model in models[1:]:
            tracks[model['id']] = np.load(OUT / 'predictions' / model['sha256'] / (row['sha256_audio'] + '.npy'))
        for model in models:
            values = tracks[model['id']]
            if values.shape != teacher.shape:
                raise ValueError('Wrong comparison track shape')
            values.astype('<f4').tofile(target / (stem + '-' + model['id'] + '.f32'))
        pcm = np.fromfile(row['audio'], '<f4')
        with wave.open(str(target / (stem + '.wav')), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())
        events = list(metrics.associate_events(row['phones'], metrics.runs(np.median(truth, axis=-1) <= .1)))
        wanted = row['kind'].removeprefix('closure-') if row['kind'].startswith('closure-') else None
        eligible = [(phone, a, b) for phone, (a, b) in events if a >= 15 and (wanted is None or phone == wanted)]
        if eligible:
            phone, a, b = eligible[0]
            a, b = int(a), int(b)
            centre = (a + b - 1) // 2
            snapshots = [dict(frame=max(0, a - 4), title=f'Before /{phone}/'),
                         dict(frame=centre, title=f'Contact /{phone}/'),
                         dict(frame=min(len(teacher) - 1, b + 4), title=f'After /{phone}/')]
        else:
            centre = int(np.median(truth, axis=-1).argmax())
            snapshots = [dict(frame=centre, title='Maximum teacher opening')]
        fig, ax = plt.subplots(figsize=(12, 2.8), layout='constrained')
        times = np.arange(len(teacher)) / 60
        ax.plot(times, np.median(truth, axis=-1) * 10, color='black', linewidth=1.8, label='Raw teacher')
        colors = ['#999999', '#9a89b5', '#2b8cbe', '#5aae61', '#d95f02', '#df65b0']
        for model, color in zip(models, colors):
            gaps = np.einsum('fc,cg->fg', tracks[model['id']], basis, optimize=False) + rest
            ax.plot(times, np.median(gaps, axis=-1) * 10, color=color, linewidth=1., label=model['title'])
        ax.axhline(1., color='#b2182b', linestyle='--', linewidth=.8)
        ax.axvline(centre / 60, color='#777777', linestyle=':', linewidth=.8)
        ax.set(xlabel='Time (s)', ylabel='Median vertical lip gap (mm)', xlim=(0, len(pcm) / 16000))
        ax.legend(ncol=3, fontsize=7)
        fig.savefig(target / (stem + '-gaps.png'), dpi=160)
        plt.close(fig)
        cases.append(dict(stem=stem, title=row['engine'].title() + ' ' + row['voice'] + ' · ' + row['kind'],
                          text=row['text'], frames=len(teacher), snapshots=snapshots))
    (target / 'manifest.json').write_text(json.dumps(dict(models=models, cases=cases, provisional=True), indent=2))
    page = OUT / 'browser/web/review.html'
    page.write_text(page.read_text().replace('repeat(5,minmax(0,1fr))', 'repeat(6,minmax(0,1fr))'))
    print('Built six predetermined clips; one-seed research preview, not a release.')


if __name__ == '__main__':
    main()

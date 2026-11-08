"""Synchronized raw-surface supervision review, never a new training label."""
import argparse
import json
import wave
from geometry import prepare, squared_error
import numpy as np
from common import APP, OUT, freeze as frozen_write


def visual_cases(rows):
    voices = sorted({r['engine'] + ':' + r['voice'] for r in rows})
    selected = [next(r for r in rows if r['condition'] == 'original' and r['kind'] == kind
                     and r['engine'] + ':' + r['voice'] == voice)
                for voice in voices for kind in ['closure-m', 'closure-b', 'closure-p']]
    for engine in ['phonon', 'supertonic']:
        for condition, kind in [('half', 'closure-p'), ('quarter', 'vowels')]:
            selected.append(next(r for r in rows if r['engine'] == engine and r['condition'] == condition and r['kind'] == kind))
    return [r['stem'] for r in selected]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    rows = {r['stem']: r for r in json.loads((OUT / 'confirmation/variants/manifest.json').read_text()) if r['split'] == 'test'}
    chosen = visual_cases(list(rows.values()))
    frozen_write(OUT / 'visual-cases.json', json.dumps(chosen, indent=2).encode())
    if args.freeze_only:
        print('Sixteen visual cases fixed by voice/category/condition before test scoring.'); return
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import sys, os
    sys.path.insert(0, str(APP / 'scripts/research/multivoice'))
    import closure_metrics
    models = [dict(id='teacher', title='Full teacher fitted to James')]
    for name, title in [('v4', 'Current v4 finetune-r4'), ('control', 'Matched ordinary fine-tuning'), ('surface', 'Dense lip-surface supervision')]:
        result = json.loads((OUT / ('evaluation-' + name + '.json')).read_text())
        models.append(dict(id=name, title=title, sha256=result['model_sha256']))
    rig = np.load(os.environ['A2F_RESEARCH_WORK'] + '/arkit-student/rig-metrics.npz')
    basis, rest = rig['gap_basis'], rig['rest'] + rig['gap_offset']
    surface_rig=prepare()
    target = OUT / 'review'; target.mkdir(exist_ok=True)
    cases = []
    for stem in chosen:
        row = rows[stem]
        with np.load(OUT / 'confirmation/variants/labels' / (stem + '.npz')) as z:
            truth, teacher = z['gaps'], z['weights']
        tracks = {'teacher': teacher, **{m['id']: np.load(OUT / 'predictions' / m['sha256'] / (row['sha256_audio'] + '.npy')) for m in models[1:]}}
        for name, values in tracks.items():
            if values.shape != teacher.shape or not np.isfinite(values).all():
                raise ValueError('Invalid review track')
            values.astype('<f4').tofile(target / (stem + '-' + name + '.f32'))
        pcm = np.fromfile(row['audio'], '<f4')
        with wave.open(str(target / (stem + '.wav')), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes((pcm * 32767).astype('<i2').tobytes())
        associated = closure_metrics.associate(row['phones'], closure_metrics.runs(np.median(truth, -1) <= .1), len(truth))
        contacts = [e for e in associated if e['status'] == 'unique' and e['phone']['phone'] == row['kind'].removeprefix('closure-')]
        centre = sum(contacts[0]['event']) // 2 if row['kind'].startswith('closure-') and contacts else int(np.median(truth, -1).argmax())
        snapshots = [dict(frame=max(0, centre - 5), title='Before'), dict(frame=centre, title='Teacher contact/opening'),
                     dict(frame=min(len(truth) - 1, centre + 5), title='After')]
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.4), layout='constrained'); ax=axes[0]
        for model in models:
            # Avoid the NumPy/Accelerate matmul flag issue in the pinned
            # training environment; use an explicit reduction for this plot.
            gaps = truth if model['id'] == 'teacher' else np.einsum('tc,ck->tk', tracks[model['id']], basis, optimize=False) + rest
            if not np.isfinite(gaps).all():
                raise ValueError('Nonfinite plotted gaps')
            title = 'Raw teacher geometry' if model['id'] == 'teacher' else model['title']
            ax.plot(np.arange(len(truth)) / 60, 10 * np.median(gaps, -1), label=title)
        ax.set(xlabel='Time (s)', ylabel='Median vertical lip gap (mm)'); ax.legend(ncol=2, fontsize=8)
        with np.load(OUT/'surface-tracks'/(row['sha256_audio']+'.npz')) as z:surface_stats=z['stats'].astype(np.float64)
        for model in models:
            values=tracks[model['id']][:,surface_rig['active']].astype(np.float64)
            error=np.maximum(squared_error(values,surface_stats,surface_rig['gram'].astype(np.float64)),0).sum(-1)
            label='Fitted teacher rig residual' if model['id']=='teacher' else model['title']
            axes[1].plot(np.arange(len(error))/60,10*np.sqrt(error),label=label)
        axes[1].set(xlabel='Time (s)',ylabel='Dense 3D lip RMS (mm)');axes[1].legend(ncol=2,fontsize=8)
        fig.savefig(target / (stem + '-gaps.png'), dpi=140); plt.close(fig)
        cases.append(dict(stem=stem, title=f'{row["engine"]} / {row["voice"]} · {row["kind"]} · {row["condition"]}',
                          text=row['text'], frames=len(truth), snapshots=snapshots))
    (target / 'manifest.json').write_text(json.dumps(dict(models=models, cases=cases, provisional=False), indent=2))
    reference = APP / 'scripts/research/compact_ablation'
    prefix = '/' + str(OUT.relative_to(APP)) + '/review/'
    script = (reference / 'review.js').read_text().replace("'./src/rig.js'", "'/src/rig.js'").replace("'./src/idle-motion.js'", "'/src/idle-motion.js'").replace('/review/', prefix)
    (OUT / 'review.js').write_text(script)
    page = (reference / 'review.html').read_text().replace('repeat(5,minmax(0,1fr))', 'repeat(4,minmax(0,1fr))')
    (OUT / 'review.html').write_text(page.replace('Matched mouth comparison', 'Dense lip surface · matched comparison'))
    print('Open /' + str(OUT.relative_to(APP)) + '/review.html')


if __name__ == '__main__':
    main()

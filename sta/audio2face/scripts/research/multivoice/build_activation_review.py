"""Show the user's /p/ regression before/after the shader fix on the same audio."""
from pathlib import Path
import json
import os
import wave
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

APP = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('A2F_ACTIVATION_OUTPUT', APP / '.local/packed-diagnostic')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()


def main():
    target = ROOT / 'review'
    target.mkdir(exist_ok=True)
    rig = np.load(WORK / 'arkit-student/rig-metrics.npz')
    basis, rest = rig['gap_basis'], rig['rest'] + rig['gap_offset']
    models = [dict(id=model + '-' + variant, title=model.upper() + (' · original WebGPU' if variant == 'original' else ' · fixed WebGPU'))
              for model in ['v3', 'v4'] for variant in ['original', 'fixed']]
    cases = []
    for engine, audio, frame in [('phonon', 'phonon-browser.f32', 56), ('supertonic', 'supertonic.f32', 89)]:
        pcm = np.fromfile(ROOT / audio, '<f4')[:96000]
        frames = (len(pcm) * 60 + 15999) // 16000
        tracks = {}
        for model in ['v3', 'v4']:
            for variant in ['original', 'fixed']:
                values = np.fromfile(ROOT / f'{engine}-{variant}-{model}.f32', '<f4').reshape(-1, 52)[:frames]
                key = model + '-' + variant
                values.astype('<f4').tofile(target / f'{engine}-{key}.f32')
                tracks[key] = values
        with wave.open(str(target / (engine + '.wav')), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())
        fig, ax = plt.subplots(figsize=(12, 2.8), layout='constrained')
        times = np.arange(frames) / 60
        for model, color in zip(models, ['#b35806', '#f1a340', '#542788', '#2b8cbe']):
            gaps = np.einsum('fc,cg->fg', tracks[model['id']], basis, optimize=False) + rest
            ax.plot(times, np.median(gaps, axis=-1) * 10, color=color,
                    linewidth=1.2, linestyle='--' if model['id'].endswith('original') else '-', label=model['title'])
        ax.axvline(frame / 60, color='#555555', linestyle=':', linewidth=.8)
        ax.set(xlabel='Time (s)', ylabel='Median vertical lip gap (mm)', xlim=(max(0, frame / 60 - .5), frame / 60 + .5))
        ax.legend(ncol=4, fontsize=8); fig.savefig(target / (engine + '-gaps.png'), dpi=160); plt.close(fig)
        cases.append(dict(stem=engine, title=engine.title() + ' · packed regression', frames=frames,
                          text='Peter and Ben packed a big blue bag, then walked to the village by the river. (First six seconds.)',
                          snapshots=[dict(frame=frame - 6, title='Before affected frames'),
                                     dict(frame=frame, title='Affected frame · original shader'),
                                     dict(frame=frame + 8, title='After affected frames')]))
    (target / 'manifest.json').write_text(json.dumps(dict(models=models, cases=cases, provisional=False), indent=2))
    reference = APP / 'scripts/research/compact_ablation'
    script = (reference / 'review.js').read_text().replace("'./src/rig.js'", "'/src/rig.js'").replace("'./src/idle-motion.js'", "'/src/idle-motion.js'")
    prefix = '/' + str(ROOT.relative_to(APP)) + '/review/'
    script = script.replace('/review/', prefix)
    (ROOT / 'packed.js').write_text(script)
    page = (reference / 'review.html').read_text().replace('repeat(5,minmax(0,1fr))', 'repeat(4,minmax(0,1fr))')
    page = page.replace('Matched mouth comparison', 'Packed /p/ · original and fixed WebGPU')
    page = page.replace('The same speech and playback clock drive every face. The reference is full Audio2Face fitted to James’s blendshapes. Compare closures, jaw opening and transitions; the head can be held steady for inspection.',
                        'Identical recorded browser speech drives every face. Only the shader activation changes; model weights are unchanged. Fixed outputs agree with native CPU inference. Hold the head steady to inspect the transient neutral-pose glitch.')
    (ROOT / 'packed.html').write_text(page.replace('./review.js', './packed.js'))
    print('Prepared', prefix.replace('/review/', '/packed.html'))


if __name__ == '__main__':
    main()

"""Freeze complete-track native references for the unchanged browser runner."""
import json
import sys
import numpy as np
import onnxruntime as ort
from prepare_corpus import APP, OUT, freeze, sha

sys.path.insert(0, str(APP / 'scripts/research/compact_ablation'))
import metrics


def main():
    selection = json.loads((OUT / 'selection-lock.json').read_text())
    rows = json.loads((OUT / 'datasets/confirmation/manifest.json').read_text())
    models = {arm: OUT / arm / 'model.onnx' for arm in ['original-short','original-long','expanded-short','expanded-long']}
    fixtures = []
    for engine in ['phonon', 'supertonic']:
        for condition in ['original', 'half', 'boost', 'quarter']:
            row = next(r for r in rows if r['engine'] == engine and r['condition'] == condition)
            fixtures.append((engine + '-' + condition, __import__('pathlib').Path(row['audio'])))
        other=next(r for r in rows if r['engine']==engine and r['condition']=='original' and r['gender']=='female')
        fixtures.append((engine+'-female-original', __import__('pathlib').Path(other['audio'])))
        name = 'phonon-browser.f32' if engine == 'phonon' else 'supertonic.f32'
        fixtures.append((engine + '-packed', APP / '.local/packed-diagnostic' / name))
    models['v4']=APP/'.local/compact-coverage-v2/coverage/runs/control-replay-s37/model.onnx'
    root = OUT / 'browser/parity'; root.mkdir(parents=True, exist_ok=True)
    cases, hashes = [], {}
    for name, audio in fixtures:
        freeze(root / (name + '.f32'), audio.read_bytes())
        hashes[name] = sha(audio)
        cases.append(dict(name=name, audio=name + '.f32', reference={}))
    for arm, path in models.items():
        if arm!='v4' and sha(path) != selection[arm]['model_sha256']:
            raise ValueError('Changed selected export')
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        for case in cases:
            track = metrics.infer(session, np.fromfile(root / case['audio'], '<f4'))
            if not np.isfinite(track).all():
                raise ValueError('Invalid native track')
            filename = case['name'] + '-' + arm + '-cpu.f32'
            freeze(root / filename, track.astype('<f4').tobytes())
            case['reference'][arm] = filename
    freeze(root / 'parity-cases.json', cases)
    freeze(root / 'fixture-lock.json', dict(source_sha256=sha(__file__), audio=hashes,
        model_sha256={arm: sha(path) for arm, path in models.items()},
        runner_sha256=sha(APP / 'src/compact-webgpu.js'),
        convolution_sha256=sha(APP / 'src/compact-conv.wgsl'),
        projection_sha256=sha(APP / 'src/compact-project.wgsl')))
    print('Frozen', len(cases), 'complete-track cases for both unchanged-topology exports.')


if __name__ == '__main__':
    main()

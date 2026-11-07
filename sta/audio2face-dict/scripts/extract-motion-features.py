"""Extract training-only motion measurements from full native Audio2Face."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from motion_features import MotionFeatures, NAMES

parser = argparse.ArgumentParser(description=__doc__)
for name in ['runner', 'model', 'rig']:
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(args.runner.resolve()))
from app.audio2face_v3 import animate, load_backend, MODEL_SHA256

rows = []
for name in ['full', 'targeted', 'subtitles']:
    for row in json.loads((root / '.local' / name / 'clips.json').read_text()):
        if row['split'] == 'train':
            rows.append(dict(row, corpus=name))
if len({row['id'] for row in rows}) != len(rows):
    raise ValueError('Duplicate training clip IDs')
spec = dict(version=1, fps=60, features=NAMES, nativeModelSHA256=MODEL_SHA256,
            trainingClips=len(rows), trainingSeconds=sum(r['samples'] / 16000 for r in rows),
            measurement='Head-relative geometry; narrowing and protrusion relative to native neutral; jaw mean downward displacement; negative nearest upper-incisor-vertex distance from three lower lip markers',
            fvCaveat='Proximity proxy, not confirmed physical lip/tooth surface contact',
            trainingManifestSHA256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            extractorSHA256=hashlib.sha256((Path(__file__).parent / 'lib/motion_features.py').read_bytes()).hexdigest())
fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
cache = root / '.local/motion-features' / fingerprint
cache.mkdir(parents=True, exist_ok=True)
(cache / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
measure = MotionFeatures(args.rig, root / 'static/assets')
infer = None
for index, row in enumerate(rows, 1):
    source = root / '.local' / row['corpus']
    count = (row['samples'] * 60 + 15999) // 16000
    native_file, fitted_file = [cache / (row['id'] + '.' + name + '.f32') for name in ['native', 'fitted']]
    size = count * len(NAMES) * 4
    if fitted_file.exists() and fitted_file.stat().st_size == size:
        pass
    else:
        fit = np.fromfile(source / (row['id'] + '.lip.arkit'), dtype='<f4')
        if fit.size != count * 52:
            raise ValueError('Incomplete fit for ' + row['id'])
        blocks = [measure.fitted(fit.reshape(-1, 52)[start:start + 30]) for start in range(0, count, 30)]
        values = np.concatenate(blocks)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite fitted features')
        temporary = fitted_file.with_suffix('.partial')
        values.tofile(temporary)
        temporary.replace(fitted_file)
    if not (native_file.exists() and native_file.stat().st_size == size):
        if infer is None:
            infer = load_backend(args.model, 'mlx')
        audio = np.fromfile(source / (row['id'] + '.pcm'), dtype='<f4')
        if len(audio) != row['samples']:
            raise ValueError('Incomplete audio for ' + row['id'])
        blocks, cursor = [], 0
        for indices, geometry in animate(audio, infer, identity='James', seed=0):
            if not np.array_equal(indices, np.arange(cursor, cursor + len(indices)) * 16000 // 60):
                raise ValueError('Native timestamps differ from 60 fps grid')
            blocks.append(measure.native(geometry))
            cursor += len(indices)
        values = np.concatenate(blocks)
        if values.shape != (count, len(NAMES)) or not np.isfinite(values).all():
            raise ValueError('Invalid native feature track')
        temporary = native_file.with_suffix('.partial')
        values.tofile(temporary)
        temporary.replace(native_file)
    print(f'{index}/{len(rows)} {row["id"]} {count} frames', flush=True)
pointer = root / '.local/motion-features/current.json'
pointer.write_text(json.dumps(dict(directory=fingerprint, **spec), indent=2) + '\n')
print(f'Completed {len(rows)} training clips in {cache}', flush=True)

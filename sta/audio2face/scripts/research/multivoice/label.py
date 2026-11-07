"""Generate consistent James labels and provenance for every new recording."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time
import numpy as np

os.environ['MLX_ENABLE_TF32'] = '0'
APP = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('A2F_MULTIVOICE_OUTPUT', APP / '.local/compact-multivoice')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()
TEACHER = Path(os.environ['A2F_TEACHER_ROOT']).resolve()
ASSETS = Path(os.environ.get('A2F_ASSET_ROOT', WORK / 'v3-model'))
sys.path.insert(0, str(TEACHER))
from app.audio2face_v3 import animate, verify_model, MODEL_SHA256

FEATURE = '/model/audio_model/encoder/layers.11/final_layer_norm/Add_1_output_0'


def feature_backend():
    """Expose the teacher's existing speech encoder, only in an offline copy."""
    verify_model(TEACHER / 'models_cache/audio2face-v3/network.onnx')
    source = (TEACHER / 'app/audio2face_v3/mlx_backend.py').read_text()
    replacements = [
        ('output_names = ("prediction", "output_latents")',
         f'output_names = ("prediction", "output_latents", {FEATURE!r})'),
        ('return values["prediction"], values["output_latents"]',
         f'return values["prediction"], values["output_latents"], values[{FEATURE!r}]'),
        ('prediction, state = graph(tensors)', 'prediction, state, features = graph(tensors)'),
        ('mx.eval(prediction, state)', 'mx.eval(prediction, state, features)'),
        ('return _host(prediction), _host(state)', 'return _host(prediction), _host(state), _host(features)'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f'Teacher instrumentation no longer matches: {old}')
        source = source.replace(old, new)
    path = OUT / 'sources/mlx_features.py'
    if path.exists() and path.read_text() != source:
        raise ValueError('Instrumented teacher changed; use a new study directory')
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('app.audio2face_v3.multivoice_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mlx_backend(TEACHER / 'models_cache/audio2face-v3/network.onnx')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parity = json.loads((OUT / 'instrumentation-parity.json').read_text())
    if parity['instrumented_backend_sha256'] != sha(OUT / 'sources/mlx_features.py'):
        raise ValueError('Teacher instrumentation parity must be rerun')
    spec = importlib.util.spec_from_file_location('fitter', OUT / 'sources/label.py')
    fitter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fitter)
    _, fit, _, gaps = fitter.prepare_fit(ASSETS)
    recipe = dict(teacher_sha256=MODEL_SHA256, identity='James', emotion='neutral', seed=0,
                  mlx_tf32=False, labeler_sha256=sha(__file__), fitter_sha256=sha(OUT / 'sources/label.py'),
                  assets={p.name: sha(p) for p in ASSETS.glob('*James*')},
                  teacher_source={str(p.relative_to(TEACHER)): sha(p)
                                  for p in (TEACHER / 'app/audio2face_v3').glob('*.py')},
                  feature_node=FEATURE, feature_sample_center=199.5, feature_hop=320,
                  feature_alignment='linear interpolation onto 60fps sample positions; f16 cache')
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    target_dir = OUT / 'labels'
    target_dir.mkdir(exist_ok=True)
    (OUT / 'label-recipe.json').write_text(json.dumps(recipe, indent=2))
    backend = None
    latest = None
    call_count = 0
    latest_start = 0
    def wrapped(inputs):
        nonlocal latest, call_count, latest_start
        latest_start = -16000 + call_count * 8000
        call_count += 1
        geometry, state, features = backend(inputs)
        if features.shape != (1, 49, 768):
            raise ValueError(f'Unexpected teacher feature shape: {features.shape}')
        # Valid-strided waveform conv features: 400-sample receptive field,
        # 320-sample hop. Match each student window's 60fps sample positions.
        positions = np.clip((np.arange(60) * 16000 // 60 - 199.5) / 320, 0, 48)
        left = positions.astype(int)
        right = np.minimum(left + 1, 48)
        fraction = (positions - left).astype(np.float32)[:, None]
        latest = features[0, left] + fraction * (features[0, right] - features[0, left])
        return geometry, state
    started = time.monotonic()
    rows = json.loads((OUT / 'manifest.json').read_text())
    for i, row in enumerate(rows):
        audio = Path(row['audio'])
        if sha(audio) != row['sha256_audio']:
            raise ValueError(f'Audio changed: {audio}')
        target = target_dir / (row['stem'] + '.npz')
        frames = (row['samples'] * 60 + 15999) // 16000
        if target.exists():
            with np.load(target) as z:
                if (str(z['recipe_sha256']) != recipe_sha or str(z['audio_sha256']) != row['sha256_audio']
                        or z['weights'].shape != (frames, 52) or z['gaps'].shape != (frames, 9)
                        or z['speech_features'].shape != (frames, 768)):
                    raise ValueError(f'Cached label changed: {target}')
            continue
        if backend is None:
            # Instrumentation parity is separately checked by verify.py before
            # the study starts; it also verifies the pinned original graph hash.
            backend = feature_backend()
        pcm = np.fromfile(audio, '<f4')
        ws, gs, fs = [], [], []
        cursor = 0
        call_count = 0
        for indices, geometry in animate(pcm, wrapped, identity='James', seed=0):
            expected = np.arange(cursor, cursor + len(indices)) * 16000 // 60
            if not np.array_equal(indices, expected):
                raise ValueError('Teacher timestamp mismatch')
            ws.append(fit(geometry))
            gs.append(gaps(geometry))
            # Invert floor(frame * 16000 / 60) using integer ceiling. This also
            # handles the partial first and last centre crops exactly.
            selected = ((indices - latest_start) * 60 + 15999) // 16000
            if np.any((selected < 15) | (selected >= 45)):
                raise ValueError('Feature/geometry crop mismatch')
            fs.append(latest[selected].astype(np.float16))
            cursor += len(indices)
        w, g = np.concatenate(ws), np.concatenate(gs)
        if w.shape != (frames, 52) or g.shape != (frames, 9) or not np.isfinite(w).all() or not np.isfinite(g).all():
            raise ValueError(f'Invalid label: {row["stem"]}')
        temporary = target.with_suffix('.part.npz')
        features = np.concatenate(fs)
        if features.shape != (frames, 768) or not np.isfinite(features).all():
            raise ValueError('Invalid speech features')
        np.savez_compressed(temporary, weights=w, gaps=g, speech_features=features,
                            recipe_sha256=recipe_sha, audio_sha256=row['sha256_audio'])
        temporary.replace(target)
        print('Labeled', i + 1, '/', len(rows), row['stem'], 'elapsed', round(time.monotonic() - started), flush=True)


if __name__ == '__main__':
    main()

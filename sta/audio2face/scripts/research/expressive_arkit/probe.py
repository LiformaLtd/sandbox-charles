"""Measure the pinned teacher's actual expression response before training."""
import os
import sys
import time

os.environ['MLX_ENABLE_TF32'] = '0'
import numpy as np
from common import *
from fit import FullFaceFit


def main():
    sys.path.insert(0, str(TEACHER))
    from app.audio2face_v3 import load_backend, animate, MODEL_SHA256
    if sha(BASELINE) != EXPECTED_BASELINE:
        raise ValueError('Unexpected released baseline')
    fitter = FullFaceFit(ASSETS,WORK/'arkit-student/james_arkit_v3.ma',APP)
    names = fitter.names
    original = load_backend(TEACHER / 'models_cache/audio2face-v3/network.onnx', 'mlx')
    manifest = SOURCE_APP / '.local/compact-multivoice/manifest.json'
    row = json.loads(manifest.read_text())[0]
    audio = Path(row['audio'])
    if sha(audio) != row['sha256_audio']:
        raise ValueError('Changed source recording')
    pcm = np.fromfile(audio, '<f4')
    records = []
    arrays = {}
    started = time.monotonic()
    for index, name in enumerate(['neutral', *EMOTIONS]):
        emotion = np.zeros((1, 30, 10), np.float32)
        if index:
            emotion[:, :, index - 1] = 1

        def infer(inputs):
            return original({**inputs, 'emotion': emotion})

        geometry = np.concatenate([g for _, g in animate(pcm, infer, identity='James', seed=0)])
        weights, extra = fitter(geometry)
        eyes = geometry[:, -4:]
        arrays[name + '_weights'] = weights
        arrays[name + '_eyes'] = eyes
        arrays[name + '_gaps'] = (fitter.model_neutral[fitter.upper,1]-fitter.model_neutral[fitter.lower,1]
            +geometry[:,:72006].reshape(-1,24002,3)[:,fitter.upper,1]
            -geometry[:,:72006].reshape(-1,24002,3)[:,fitter.lower,1])
        records.append(dict(emotion=name, coefficient_min=weights.min(0).tolist(),
                            coefficient_max=weights.max(0).tolist(),
                            eye_min=eyes.min(0).tolist(), eye_max=eyes.max(0).tolist()))
        print(name, 'brows', np.round(weights[:, 41:46].max(0), 3),
              'eye range', np.round(np.ptp(eyes, axis=0), 3),
              'elapsed', round(time.monotonic()-started, 1), flush=True)
    with np.load(ASSETS / 'bs_skin_James.npz') as rig:
        channel_audit = [dict(index=i, name=n, skin_delta_norm=float(np.linalg.norm(rig[n])),
                              previous_active=bool(i in np.flatnonzero(json.loads(
                                  (ASSETS / 'bs_skin_config_James.json').read_text()
                              )['blendshape_params']['bsSolveActivePoses']))) for i, n in enumerate(names)]
    np.savez_compressed(OUT / 'teacher-probe.npz', **arrays)
    freeze(OUT / 'teacher-probe.json', dict(teacher_sha256=MODEL_SHA256,
        baseline_sha256=sha(BASELINE), audio_sha256=sha(audio), audio=str(audio),
        source_sha256=sha(__file__), fitter_sha256=sha(Path(__file__).with_name('fit.py')),
        note='Full-face research fitter with declared gaze and tongue proxy calibrations.',
        channels=names, channel_audit=channel_audit, records=records))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    main()

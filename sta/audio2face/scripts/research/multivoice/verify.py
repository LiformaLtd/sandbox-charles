"""Check label instrumentation against the untouched full teacher before labeling."""
import json
import numpy as np
import label
from app.audio2face_v3 import animate, load_backend


def main():
    # The standard backend verifies the original graph SHA-256.
    reference = load_backend(label.TEACHER / 'models_cache/audio2face-v3/network.onnx', 'mlx')
    instrumented = label.feature_backend()
    rows = json.loads((label.OUT / 'planned.json').read_text())
    first = json.loads((label.OUT / 'synthesis' / (rows[0]['stem'] + '.json')).read_text())
    pcm = np.fromfile(first['audio'], '<f4')[:12000]
    truth = np.concatenate([g for _, g in animate(pcm, reference, identity='James', seed=0)])
    captured = []
    def wrapped(inputs):
        geometry, state, features = instrumented(inputs)
        assert features.shape == (1, 49, 768) and np.isfinite(features).all()
        captured.append(features)
        return geometry, state
    prediction = np.concatenate([g for _, g in animate(pcm, wrapped, identity='James', seed=0)])
    np.testing.assert_allclose(prediction, truth, atol=2e-4, rtol=1e-4)
    result = dict(geometry_values=truth.size, max_abs_difference=float(np.max(np.abs(truth - prediction))),
                  feature_shape=list(captured[-1].shape), teacher_sha256=label.MODEL_SHA256,
                  instrumented_backend_sha256=label.sha(label.OUT / 'sources/mlx_features.py'))
    (label.OUT / 'instrumentation-parity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

"""Fit an acoustic/context ranking target using only dictionary training clips."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
packed = root / 'static/dictionary'
data = json.loads((packed / 'index.json').read_text())
if not all('acoustics' in clip for clip in data['clips']):
    raise ValueError('Pack the training corpus with --acoustics first')
weights = np.fromfile(packed / data['weightsFile'], dtype='<f4').reshape(-1, 52)
labels = sorted({p['phone'] for c in data['clips'] for p in c['phones']})
neutral = np.array(data['lipRig']['neutral'])[[2, 4, 6]]
deltas = np.array(data['lipRig']['deltas'])[:, [2, 4, 6]]
x, y = [], []
for clip in data['clips']:
    for i, phone in enumerate(clip['phones']):
        audio = clip['acoustics'][i]
        features = audio[:6] + [audio[6] * max(0, 2 * audio[7] - 1), audio[7], np.log(phone['end'] - phone['start'])]
        for at in [i - 1, i, i + 1]:
            name = clip['phones'][at]['phone'] if 0 <= at < len(clip['phones']) else None
            features += [int(name == label) for label in labels]
        x.append(features)
        frame = np.clip((phone['start'] + phone['end']) / 2 * 60, 0, clip['frameCount'] - 1)
        before = int(frame) + clip['frameOffset']
        after = min(int(frame) + 1, clip['frameCount'] - 1) + clip['frameOffset']
        pose = weights[before] + (weights[after] - weights[before]) * (frame % 1)
        y.append(neutral + pose @ deltas)
x, y = np.array(x), np.array(y)
mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.1)
x = (x - mean) / scale
bias = y.mean(axis=0)
coefficients = np.linalg.solve(x.T @ x + 10 * np.eye(x.shape[1]), x.T @ (y - bias))
model = dict(version=1, labels=labels, mean=mean.tolist(), scale=scale.tolist(), weights=coefficients.tolist(), bias=bias.tolist(), trainingClips=len(data['clips']), trainingPhones=len(x), ridge=10, weightsFile=data['weightsFile'])
(root / '.local/selection-model.json').write_text(json.dumps(model) + '\n')
print(f'Trained selection targets on {len(data["clips"])} clips, {len(x)} phones; training RMS {np.sqrt(np.mean((x @ coefficients + bias - y)**2)):.5f}')

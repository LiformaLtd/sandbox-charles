"""Training-only test: does loudness change the teacher less than the student?"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
import label
from app.audio2face_v3 import animate, load_backend


def main():
    out, work = label.OUT, label.WORK
    rows = json.loads((out / 'manifest.json').read_text())
    train = [r for r in rows if r['split'] == 'train']
    voices = sorted({r['engine'] + ':' + r['voice'] for r in train})
    selected = [next(r for r in train if r['engine'] + ':' + r['voice'] == voice and r['kind'] == kind)
                for voice in voices for kind in ['closure-m', 'vowels']]
    spec = importlib.util.spec_from_file_location('fit_gain', out / 'sources/label.py')
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    _, _, _, geometry_gaps = fit.prepare_fit(label.ASSETS)
    teacher = load_backend(label.TEACHER / 'models_cache/audio2face-v3/network.onnx', 'mlx')
    rig = np.load(work / 'arkit-student/rig-metrics.npz')
    basis, rest = rig['gap_basis'], rig['rest'] + rig['gap_offset']
    sys.path.insert(0, str(label.APP / 'scripts/research/compact_ablation'))
    import metrics
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    models = {name: ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
              for name, path in [('v3', work / 'arkit-student-v3/james-arkit-student-v3.onnx'),
                                 ('multivoice', out / 'runs/control-s37/model.onnx')]}
    details = []
    for row in selected:
        pcm = np.fromfile(row['audio'], '<f4')
        with np.load(out / 'labels' / (row['stem'] + '.npz')) as z:
            reference = z['gaps']
        original = {n: metrics.infer(model, pcm) @ basis + rest for n, model in models.items()}
        for gain in [.5, 2.]:
            scaled = pcm * gain  # No clipping: isolate gain, not distortion.
            truth = np.concatenate([geometry_gaps(g) for _, g in animate(scaled, teacher, identity='James', seed=0)])
            record = dict(stem=row['stem'], gain=gain, frames=len(truth), peak=float(np.abs(scaled).max()),
                          teacher_delta_rmse=float(np.mean((truth - reference) ** 2) ** .5), students={})
            for n, model in models.items():
                pred = metrics.infer(model, scaled) @ basis + rest
                record['students'][n] = dict(delta_rmse=float(np.mean((pred - original[n]) ** 2) ** .5),
                                             error_against_scaled_teacher=float(np.mean((pred - truth) ** 2) ** .5))
            details.append(record)
        print('Gain-tested', row['stem'], flush=True)
    summary = {}
    for gain in [.5, 2.]:
        a = [d for d in details if d['gain'] == gain]
        weighted = lambda values: float(np.sqrt(np.average(np.square(values), weights=[d['frames'] for d in a])))
        summary[str(gain)] = dict(teacher_delta_rmse=weighted([d['teacher_delta_rmse'] for d in a]),
                                  max_teacher_clip_delta=max(d['teacher_delta_rmse'] for d in a),
                                  students={n: dict(delta_rmse=weighted([d['students'][n]['delta_rmse'] for d in a]),
                                                    error_against_scaled_teacher=weighted([d['students'][n]['error_against_scaled_teacher'] for d in a])) for n in models})
    result = dict(summary=summary, details=details, units='centimetres',
                  policy='Training voices/texts only; no clipping; threshold .005 cm worst-clip teacher RMS before considering invariant-label augmentation')
    (out / 'gain-diagnostic.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

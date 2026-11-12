"""Verify capacity is the intervention, including identical sampled minibatches."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import numpy as np
import torch
from prepare import APP, OUT, sha
from initialization import initialize_student

if os.environ.get('A2F_ONNX_SITE_PACKAGES'):
    sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        if sha(APP / path) != digest:
            raise ValueError('Frozen capacity source changed: ' + path)
    reports, checkpoints = {}, {}
    for arm, width in [('control96', 96), ('independent128', 128)]:
        root = OUT / arm
        if sha(root / 'manifest.json') != protocol['manifest_sha256']:
            raise ValueError('Training data differs')
        run = root / 'runs/control-replay-s37'
        report = json.loads((run / 'training.json').read_text())
        if report['recipe']['baseline_sha256'] != protocol['initialization_sha256'] or report['recipe']['config']['width'] != width:
            raise ValueError('Wrong initialization or width')
        if len(report['history']) != 24 or report['steps_per_epoch'] != 31 or sha(run / 'model.onnx') != report['model_sha256']:
            raise ValueError('Incomplete or changed export')
        reports[arm] = report
        checkpoints[arm] = torch.load(run / 'checkpoint.pt', weights_only=False)
    a, b = reports.values()
    for name in ['train_sha256', 'dev_sha256', 'clean_replay_fingerprints']:
        if a['recipe'][name] != b['recipe'][name]:
            raise ValueError('Unmatched data/replay: ' + name)
    configs = [{k: v for k, v in report['recipe']['config'].items() if k != 'width'} for report in reports.values()]
    if configs[0] != configs[1] or not torch.equal(checkpoints['control96']['rng'], checkpoints['independent128']['rng']):
        raise ValueError('Unmatched config or minibatch sampler state')
    initial_delta = max(abs(x[k] - y[k]) for x, y in zip(a['initial_dev'], b['initial_dev']) for k in ['lip_gap_rmse', 'contact_f1'])
    if initial_delta > 2e-5:
        raise ValueError('Initial function metrics differ unexpectedly')
    graph = onnx.load(protocol['initialization'])
    values = {t.name: numpy_helper.to_array(t).copy() for t in graph.graph.initializer}
    spec = importlib.util.spec_from_file_location('capacity_audit_model', OUT / 'independent128/sources/model.py')
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    torch.manual_seed(37)
    student = base.Student(128, values['student.mean'], values['student.scale'], values['student.channel_map'].argmax(1))
    initialize_student(student, values)
    original_extra = student.encoder.weight[96:].detach().clone()
    selected = torch.load(OUT / 'independent128/runs/control-replay-s37/student.pt', weights_only=False)
    change = float((selected['encoder.weight'][96:] - original_extra).norm() / original_extra.norm())
    selection = {arm: {k: report[k] for k in ['best_epoch', 'best_selection', 'model_sha256', 'recipe_sha256', 'parameters', 'bytes']}
                 for arm, report in reports.items()}
    target = OUT / 'selection.json'
    if target.exists() and json.loads(target.read_text()) != selection:
        raise ValueError('Previously frozen selection differs')
    target.write_text(json.dumps(selection, indent=2))
    result = dict(initial_development_max_delta=initial_delta, data_and_replay_identical=True,
                  minibatch_sampler_state_identical=True, updates_per_arm=744,
                  selected_new_encoder_relative_weight_change=change,
                  selected_extra_output_weight_norm=float(selected['output.weight'][:, 96:].norm()), selected=selection)
    (OUT / 'training-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

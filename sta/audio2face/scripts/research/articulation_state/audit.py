"""Lock exports before confirmation scoring and audit the controlled intervention."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import torch
from prepare import APP, OUT, sha, freeze
from states import articulation_targets

if os.environ.get('A2F_ONNX_SITE_PACKAGES'):
    sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        if sha(APP / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    reports, checkpoints, state_counts = {}, {}, {}
    for arm in ['control', 'state']:
        root = OUT / arm
        if sha(root / 'manifest.json') != protocol['manifest_sha256']:
            raise ValueError('Unmatched manifest')
        run = root / 'runs' / f'{arm}-replay-s37'
        report = json.loads((run / 'training.json').read_text())
        if len(report['history']) != 24 or report['steps_per_epoch'] != 31:
            raise ValueError('Incomplete training')
        if report['recipe']['baseline_sha256'] != protocol['initialization_sha256']:
            raise ValueError('Wrong initialization')
        if sha(run / 'model.onnx') != report['model_sha256']:
            raise ValueError('Changed model')
        graph = onnx.load(run / 'model.onnx')
        baseline = onnx.load(protocol['initialization'])
        if [(v.op_type, list(v.input), list(v.output)) for v in graph.graph.node] != [(v.op_type, list(v.input), list(v.output)) for v in baseline.graph.node]:
            raise ValueError('Runtime graph differs')
        if report['bytes'] != 2251827 or report['parameters'] != 331047:
            raise ValueError('Unexpected deployed size/capacity')
        reports[arm] = report
        checkpoints[arm] = torch.load(run / 'checkpoint.pt', weights_only=False)
        cache = root / ('features-train-' + report['recipe']['train_sha256'][:16] + '.npz')
        with np.load(cache) as z:
            targets = articulation_targets(torch.from_numpy(z['gaps'].copy()), torch.from_numpy(z['valid'].copy()))
        state_counts[arm] = torch.bincount(targets[targets >= 0], minlength=4).tolist()
    a, b = reports.values()
    for name in ['train_sha256', 'dev_sha256', 'clean_replay_fingerprints']:
        if a['recipe'][name] != b['recipe'][name]:
            raise ValueError('Unmatched data/replay')
    configs = [{k: v for k, v in r['recipe']['config'].items() if k != 'arm'} for r in reports.values()]
    if configs[0] != configs[1] or not torch.equal(checkpoints['control']['rng'], checkpoints['state']['rng']):
        raise ValueError('Unmatched recipe or sampled minibatches')
    if a['initial_dev'] != b['initial_dev'] or state_counts['control'] != state_counts['state']:
        raise ValueError('Different initial function or target distribution')
    selection = {arm: {k: r[k] for k in ['best_epoch', 'best_selection', 'model_sha256', 'recipe_sha256', 'parameters', 'bytes']} for arm, r in reports.items()}
    freeze(OUT / 'selection.json', selection)
    result = dict(selection=selection, updates_per_arm=744, identical_initial_development=True,
                  identical_data_replay_and_sampler=True, runtime_graph_identical=True,
                  primary_training_state_counts=state_counts,
                  final_auxiliary_head_difference_norm=float((checkpoints['state']['model']['state_head.weight'] - checkpoints['control']['model']['state_head.weight']).norm()))
    (OUT / 'training-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

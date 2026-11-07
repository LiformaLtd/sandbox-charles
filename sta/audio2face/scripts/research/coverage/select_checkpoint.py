"""Freeze development-selected exports before opening held-out scores."""
import json
from prepare import APP, OUT, freeze, sha


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    selected = {}
    for arm in ['control', 'coverage']:
        root = OUT / arm / 'runs/control-replay-s37'
        log = json.loads((root / 'training.json').read_text())
        if log['recipe']['baseline_sha256'] != protocol['initialization_sha256']:
            raise ValueError('Initialization changed')
        expected_code = protocol['source_sha256']['scripts/research/multivoice/train.py']
        if log['recipe']['code_sha256'] != expected_code or sha(APP / 'scripts/research/multivoice/train.py') != expected_code:
            raise ValueError('Training source changed')
        if log['steps_per_epoch'] != 31 or len(log['history']) != 24 or log['parameters'] != 331047 or log['bytes'] != 2251827:
            raise ValueError('Incomplete or changed training budget/topology')
        if sha(root / 'model.onnx') != log['model_sha256']:
            raise ValueError('Changed selected export')
        selected[arm] = {k: log[k] for k in ['best_epoch', 'best_selection', 'model_sha256', 'recipe_sha256', 'parameters', 'bytes']}
    freeze(OUT / 'selection.json', selected)
    print(json.dumps(selected, indent=2))


if __name__ == '__main__':
    main()

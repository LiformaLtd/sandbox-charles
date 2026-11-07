"""Merge modulo-96 channel replicas from a Net2Wider student into width 96."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import onnx
from onnx import numpy_helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    baseline = args.baseline or Path(os.environ['A2F_RESEARCH_WORK']) / 'arkit-student-v3/james-arkit-student-v3.onnx'
    source = onnx.load(args.source)
    values = {t.name: numpy_helper.to_array(t).copy() for t in source.graph.initializer}
    width = values['student.encoder.weight'].shape[0]
    if width < 96 or width % 96:
        raise ValueError('This folding implementation requires a multiple of 96 hidden channels.')
    copies = width // 96
    graph = onnx.load(baseline)
    if [n.op_type for n in source.graph.node] != [n.op_type for n in graph.graph.node]:
        raise ValueError('Unsupported graph topology for this folding recipe.')
    conv_attributes = lambda model: [[a.SerializeToString() for a in n.attribute]
                                     for n in model.graph.node if n.op_type == 'Conv']
    if conv_attributes(source) != conv_attributes(graph):
        raise ValueError('Convolution timing/structure differs from the baseline.')
    diagnostic = {}
    for tensor in graph.graph.initializer:
        name = tensor.name
        if not name.startswith('student.') or not name.endswith(('.weight', '.bias')):
            continue
        value = values[name]
        if name == 'student.encoder.weight':
            grouped = value.reshape(copies, 96, *value.shape[1:])
            value = grouped.mean(0)
            differences = grouped - value[None]
            diagnostic['encoder_replica_relative_rms'] = float(np.sqrt(np.mean(differences ** 2)) / max(1e-12, np.sqrt(np.mean(grouped ** 2))))
        elif name == 'student.encoder.bias':
            value = value.reshape(copies, 96).mean(0)
        elif name.startswith('student.blocks.'):
            if name.endswith('.weight'):
                value = value.reshape(copies, 96, copies, 96, value.shape[-1]).sum(2).mean(0)
            else:
                value = value.reshape(copies, 96).mean(0)
        elif name == 'student.output.weight':
            value = value.reshape(value.shape[0], copies, 96, 1).sum(1)
        expected = tuple(tensor.dims)
        if value.shape != expected:
            raise ValueError(f'{name}: got {value.shape}, expected {expected}')
        tensor.CopyFrom(numpy_helper.from_array(value.astype(np.float32), name))
    # All fixed buffers must agree; never silently fold a different frontend/map.
    for tensor in graph.graph.initializer:
        if tensor.name in ('filters','mel') or (tensor.name.startswith('student.') and not tensor.name.endswith(('.weight', '.bias'))):
            np.testing.assert_array_equal(numpy_helper.to_array(tensor), values[tensor.name])
    onnx.checker.check_model(graph)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, args.destination)
    diagnostic.update(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      sha256=hashlib.sha256(args.destination.read_bytes()).hexdigest(),
                      source_width=width, target_width=96, bytes=args.destination.stat().st_size)
    args.destination.with_suffix('.fold.json').write_text(json.dumps(diagnostic, indent=2))
    print(json.dumps(diagnostic, indent=2))


if __name__ == '__main__':
    main()

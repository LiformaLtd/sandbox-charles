"""Generate isolated WebGPU prototype, retaining original frontend/conv/projection."""
from common import APP, OUT, HERE, freeze, sha


def main():
    root=OUT/'browser';root.mkdir(exist_ok=True)
    import onnx, json
    names = {}
    for arm in ['control', 'temporal']:
        graph = onnx.load(OUT / arm / 'model.onnx').graph
        names[arm] = [next(v.input[1] for v in graph.node if v.name == '/decoder/' + layer + '/MatMul') for layer in ['input', 'output']]
    source=(APP/'src/compact-webgpu.js').read_text()
    for name in ['compact-conv.wgsl','compact-project.wgsl','compact-mel.js']:
        source=source.replace("'./"+name, "'/src/"+name)
    source="const matrixNames = " + json.dumps(names) + ";\n" + source
    source="import temporal from './temporal.wgsl?raw';\n"+source
    source=source.replace('createCompactWebGpu(adapter, data, onLost)', "createCompactWebGpu(adapter, data, onLost, mode = 'temporal')")
    source=source.replace('const [convPipeline, projectPipeline]', 'const [convPipeline, projectPipeline, temporalPipeline]')
    source=source.replace('createPipeline(conv), createPipeline(project)', 'createPipeline(conv), createPipeline(project), createPipeline(temporal)')
    source=source.replace('39 * (hidden + 4);', '39 * (hidden + 4) + 13159;')
    source=source.replace('const projection = bind(', 'const projectOffset = offset;\n    const projection = bind(')
    source=source.replace('device.lost.then(onLost);', '''const decoderOffset = offset;
    append(read('decoder.center', 96)); append(read('decoder.spread', 96));
    const transpose = (v, rows, cols) => Float32Array.from({ length: rows * cols }, (_, i) => v[(i % cols) * rows + Math.floor(i / cols)]);
    append(transpose(read(matrixNames[mode][0], 12288), 16, 768)); append(read('decoder.input.bias', 16));
    append(transpose(read(matrixNames[mode][1], 624), 39, 16)); append(read('decoder.output.bias', 39));
    if (offset !== weightCount) throw Error('Packed temporal weight count mismatch');
    const decoded = make(result.size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const history = make(672 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const params = uniform([decoderOffset, 15, 30, mode === 'temporal' ? 1 : 0, projectOffset]);
    const temporalGroup = bind(temporalPipeline, [a, model, result, decoded, history, params]);
    device.lost.then(onLost);''')
    source=source.replace('return async function runCompact(pcm) {', '''async function runCompact(pcm, { first = 15, count = 30 } = {}) {
      if (!Number.isInteger(first) || !Number.isInteger(count) || first < 15 || count < 0 || count > 30 || first + count > 45) throw Error('Invalid emitted crop');
      device.queue.writeBuffer(params, 0, new Uint32Array([decoderOffset, first, count, mode === 'temporal' ? 1 : 0, projectOffset]));''')
    source=source.replace('pass.end();', '''pass.setPipeline(temporalPipeline);
      pass.setBindGroup(0, temporalGroup);
      pass.dispatchWorkgroups(1);
      pass.end();''')
    source=source.replace('encoder.copyBufferToBuffer(result,', 'encoder.copyBufferToBuffer(decoded,')
    source=source.replace('''    };
  } catch (error)''', '''    }
    runCompact.reset = () => device.queue.writeBuffer(history, 0, new Float32Array(672));
    runCompact.dispose = () => device.destroy();
    return runCompact;
  } catch (error)''')
    freeze(root/'temporal-runner.js',source.encode())
    freeze(root/'control-runner.js',b"import {createCompactWebGpu as create} from './temporal-runner.js';\nexport const createCompactWebGpu = (a,b,c) => create(a,b,c,'control');\n")
    freeze(root/'temporal.wgsl',(HERE/'temporal.wgsl').read_bytes())
    freeze(root/'source-lock.json',dict(source_sha256=sha(__file__), runner_sha256=sha(root/'temporal-runner.js'),
        shader_sha256=sha(root/'temporal.wgsl'), matrix_initializer_names=names, baseline_runner_sha256=sha(APP/'src/compact-webgpu.js'),
        conv_sha256=sha(APP/'src/compact-conv.wgsl'),project_sha256=sha(APP/'src/compact-project.wgsl'),frontend_sha256=sha(APP/'src/compact-mel.js')))
    print('Prepared research-only third-pipeline decoder; production runtime unchanged.')


if __name__=='__main__':main()

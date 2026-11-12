import protobuf from 'protobufjs/minimal.js';
import conv from '/src/compact-conv.wgsl?raw';
import project from './residual-project.wgsl?raw';
import { createLogMel } from '/src/compact-mel.js';

const frames = 101;
const hidden = 96;
const dilations = [1, 2, 4, 8, 16, 1, 2, 4];

// Read the float initializers from the existing, versioned v2 ONNX export.
// Compact v2 and v3 share the topology implemented in the kernels.
function readTensors(data) {
  function readMessage(bytes, visit) {
    const reader = protobuf.Reader.create(bytes);
    while (reader.pos < reader.len) {
      const tag = reader.uint32();
      if (!visit(tag >>> 3, reader)) reader.skipType(tag & 7);
    }
  }
  const tensors = new Map();
  readMessage(new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength), (field, model) => {
    if (field !== 7) return false; // ModelProto.graph
    readMessage(model.bytes(), (graphField, graph) => {
      if (graphField !== 5) return false; // GraphProto.initializer
      let name, raw, type;
      readMessage(graph.bytes(), (tensorField, tensor) => {
        if (tensorField === 2) type = tensor.uint32();
        else if (tensorField === 8) name = tensor.string();
        else if (tensorField === 9) raw = tensor.bytes();
        else return false;
        return true;
      });
      if (type === 1 && name && raw && raw.byteLength % 4 === 0) {
        const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
        tensors.set(name, Float32Array.from({ length: raw.byteLength / 4 }, (_, i) => view.getFloat32(i * 4, true)));
      }
      return true;
    });
    return true;
  });
  return (name, length) => {
    const tensor = tensors.get(name);
    if (tensor?.length !== length) throw new Error(`Unexpected Compact ARKit tensor: ${name}.`);
    return tensor;
  };
}

export async function createCompactWebGpu(adapter, data, onLost) {
  const read = readTensors(data);
  const logMel = createLogMel(read('filters', 514 * 400).slice(0, 400), read('mel', 64 * 257));
  const device = await adapter.requestDevice();
  try {
    const make = (size, usage) => device.createBuffer({ size, usage });
    const input = make(64 * frames * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const states = Array.from({ length: 3 }, () => make(hidden * frames * 4, GPUBufferUsage.STORAGE));
    const result = make(60 * 52 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = make(result.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const createPipeline = (code) => device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    const [convPipeline, projectPipeline] = await Promise.all([createPipeline(conv), createPipeline(project)]);
    const bind = (pipeline, buffers) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    function uniform(values) {
      const data = new Uint32Array(values);
      const buffer = make(data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    }

    const weightCount = hidden * (64 * 5 + 1 + dilations.length * (hidden * 4 + 2)) + 39 * (hidden + 4) + 16935;
    const model = make(weightCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    let offset = 0;
    function append(data) {
      device.queue.writeBuffer(model, offset * 4, data);
      offset += data.length;
    }
    const bindings = [];
    function add(name, source, residual, target, inSize, taps, dilation, kind) {
      const params = uniform([inSize, hidden, taps, dilation, offset, kind]);
      append(read(`student.${name}.weight`, hidden * inSize * taps));
      append(read(`student.${name}.bias`, hidden));
      bindings.push(bind(convPipeline, [source, model, residual, target, params]));
    }

    let [a, b, temp] = states;
    add('encoder', input, input, a, 64, 5, 1, 0);
    for (const [i, dilation] of dilations.entries()) {
      add(`blocks.${i}.conv`, a, a, temp, hidden, 3, dilation, 0);
      add(`blocks.${i}.mix`, temp, a, b, hidden, 1, 1, 1);
      [a, b] = [b, a];
    }
    const projection = bind(projectPipeline, [a, model, result, uniform([offset, read('student.residual_kind', 1)[0]])]);
    append(read('student.output.weight', 39 * hidden));
    append(read('student.output.bias', 39));
    append(read('student.mean', 39));
    append(read('student.scale', 39));
    const channelMap = read('student.channel_map', 39 * 52);
    const channels = Float32Array.from({ length: 39 }, (_, i) => channelMap.subarray(i * 52, (i + 1) * 52).indexOf(1));
    if (channels.includes(-1) || new Set(channels).size !== channels.length) {
      throw new Error('Unexpected Compact ARKit channel map.');
    }
    append(channels);
    append(read('student.feature_scale', 96));
    append(read('student.head_in.weight', 96 * 135));
    append(read('student.head_in.bias', 96));
    append(read('student.head_out.weight', 39 * 96));
    append(read('student.head_out.bias', 39));
    if (offset !== weightCount) throw Error('Residual head packing mismatch.');
    device.lost.then(onLost);

    return async function runCompact(pcm) {
      device.queue.writeBuffer(input, 0, logMel(pcm));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(convPipeline);
      for (const group of bindings) {
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(7, 12);
      }
      pass.setPipeline(projectPipeline);
      pass.setBindGroup(0, projection);
      pass.dispatchWorkgroups(60, 1);
      pass.end();
      encoder.copyBufferToBuffer(result, 0, readback, 0, result.size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      try {
        return new Float32Array(readback.getMappedRange()).slice();
      } finally {
        readback.unmap();
      }
    };
  } catch (error) {
    device.destroy();
    throw error;
  }
}

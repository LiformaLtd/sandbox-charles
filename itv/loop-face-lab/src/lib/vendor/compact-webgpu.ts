import protobuf from 'protobufjs/minimal.js';
import conv from './compact-conv.wgsl?raw';
import project from './compact-project.wgsl?raw';
import { createLogMel } from './compact-mel';

const frames = 101;
const hidden = 96;
const dilations = [1, 2, 4, 8, 16, 1, 2, 4];

// Read the float initializers from the existing, versioned v2 ONNX export.
// Compact v2 and v3 share the topology implemented in the kernels.
function readTensors(data: ArrayBuffer) {
  function readMessage(bytes: Uint8Array, visit: (field: number, reader: protobuf.Reader) => boolean) {
    const reader = protobuf.Reader.create(bytes);
    while (reader.pos < reader.len) {
      const tag = reader.uint32();
      if (!visit(tag >>> 3, reader)) reader.skipType(tag & 7);
    }
  }
  const tensors = new Map<string, Float32Array>();
  readMessage(new Uint8Array(data), (field, model) => {
    if (field !== 7) return false; // ModelProto.graph
    readMessage(model.bytes(), (graphField, graph) => {
      if (graphField !== 5) return false; // GraphProto.initializer
      const info: { name?: string; raw?: Uint8Array; type?: number } = {};
      readMessage(graph.bytes(), (tensorField, tensor) => {
        if (tensorField === 2) info.type = tensor.uint32();
        else if (tensorField === 8) info.name = tensor.string();
        else if (tensorField === 9) info.raw = tensor.bytes();
        else return false;
        return true;
      });
      if (info.type === 1 && info.name && info.raw && info.raw.byteLength % 4 === 0) {
        const view = new DataView(info.raw.buffer as ArrayBuffer, info.raw.byteOffset, info.raw.byteLength);
        tensors.set(info.name, Float32Array.from({ length: info.raw.byteLength / 4 }, (_, i) => view.getFloat32(i * 4, true)));
      }
      return true;
    });
    return true;
  });
  return (name: string, length: number) => {
    const tensor = tensors.get(name);
    if (tensor?.length !== length) throw new Error(`Unexpected Compact ARKit tensor: ${name}.`);
    return tensor;
  };
}

export async function createCompactWebGpu(adapter: GPUAdapter, data: ArrayBuffer, onLost: () => void) {
  const read = readTensors(data);
  const logMel = createLogMel(read('filters', 514 * 400).slice(0, 400), read('mel', 64 * 257));
  const device = await adapter.requestDevice();
  try {
    const make = (size: number, usage: GPUBufferUsageFlags) => device.createBuffer({ size, usage });
    const input = make(64 * frames * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const states = Array.from({ length: 3 }, () => make(hidden * frames * 4, GPUBufferUsage.STORAGE));
    const result = make(60 * 52 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = make(result.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const createPipeline = (code: string) => device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    const [convPipeline, projectPipeline] = await Promise.all([createPipeline(conv), createPipeline(project)]);
    const bind = (pipeline: GPUComputePipeline, buffers: GPUBuffer[]) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    function uniform(values: number[]) {
      const data = new Uint32Array(values);
      const buffer = make(data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    }

    const weightCount = hidden * (64 * 5 + 1 + dilations.length * (hidden * 4 + 2)) + 39 * (hidden + 4);
    const model = make(weightCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    let offset = 0;
    function append(data: Float32Array) {
      device.queue.writeBuffer(model, offset * 4, data);
      offset += data.length;
    }
    const bindings: GPUBindGroup[] = [];
    function add(name: string, source: GPUBuffer, residual: GPUBuffer, target: GPUBuffer, inSize: number, taps: number, dilation: number, kind: number) {
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
    const projection = bind(projectPipeline, [a, model, result, uniform([offset])]);
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
    device.lost.then(onLost);

    const runCompact = async function (pcm: Float32Array) {
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
      pass.dispatchWorkgroups(8, 5);
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
    return Object.assign(runCompact, { dispose: () => device.destroy() });
  } catch (error) {
    device.destroy();
    throw error;
  }
}

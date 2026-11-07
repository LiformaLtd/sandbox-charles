struct Params {
  inSize: u32,
  outSize: u32,
  taps: u32,
  dilation: u32,
  offset: u32,
  kind: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let t = id.x;
  let c = id.y;
  if (t >= 101 || c >= params.outSize) { return; }

  let bias = params.offset + params.outSize * params.inSize * params.taps;
  var acc = weights[bias + c];
  for (var i = 0u; i < params.inSize; i++) {
    for (var k = 0u; k < params.taps; k++) {
      let at = i32(t) + (i32(k) - i32(params.taps / 2)) * i32(params.dilation);
      if (at >= 0 && at < 101) {
        let weight = params.offset + (c * params.inSize + i) * params.taps + k;
        acc += input[i * 101 + u32(at)] * weights[weight];
      }
    }
  }
  if (params.kind == 0) {
    // The encoder and dilated convolutions use the v2 GELU approximation.
    acc = 0.5 * acc * (1 + tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)));
  } else {
    // The 1x1 convolutions mix the hidden channels and add the residual.
    acc = residual[c * 101 + t] + 0.5 * acc;
  }
  result[c * 101 + t] = acc;
}

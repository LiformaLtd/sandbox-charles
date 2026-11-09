@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> offset: u32;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let t = id.x;
  let c = id.y;
  if (t >= 60 || c >= 39) { return; }

  // Resample 100 Hz features at the 60 fps animation sample positions.
  let pos = f32(t * 16000u / 60u) / 160.0;
  let low = u32(pos);
  let fraction = pos - f32(low);
  let bias = offset + 39u * 96u + c;
  var acc = weights[bias];
  for (var i = 0u; i < 96u; i++) {
    let first = input[i * 101u + low];
    let second = input[i * 101u + low + 1u];
    let feature = first + fraction * (second - first);
    acc += feature * weights[offset + c * 96u + i];
  }
  // The packed projection stores bias, mean, scale and ARKit index in order.
  let channel = u32(weights[bias + 3u * 39u]);
  let scale = weights[bias + 2u * 39u];
  let mean = weights[bias + 39u];
  result[t * 52u + channel] = clamp(acc * scale + mean, 0.0, 1.0);
}

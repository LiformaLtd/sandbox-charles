import { modelAssetUrl } from './model-assets.js';

export async function loadBlendshapeRig(sizes) {
  const response = await fetch(modelAssetUrl('/assets/james-blendshapes.bin'));
  if (!response.ok) throw new Error(`Could not load the James blendshapes (HTTP ${response.status}).`);
  const binary = await response.arrayBuffer();
  const view = new DataView(binary);
  if (binary.byteLength < 16 || view.getUint32(0, false) !== 0x4a425331) {
    throw new Error('Invalid James blendshape data.');
  }
  let offset = 4;
  const groups = sizes.map((size) => {
    const count = view.getUint32(offset, true);
    offset += 4;
    const targets = Array.from({ length: count }, () => {
      const channel = view.getUint32(offset, true);
      const vertices = view.getUint32(offset + 4, true);
      const scale = view.getFloat32(offset + 8, true);
      offset += 12;
      const points = new Int16Array(binary, offset, vertices * 4);
      offset += points.byteLength;
      return { channel, scale, points };
    });
    return { deltas: new Float32Array(size), targets };
  });
  if (offset !== binary.byteLength) throw new Error('Invalid James blendshape data length.');

  return (weights) => groups.map(({ deltas, targets }) => {
    deltas.fill(0);
    for (const { channel, scale, points } of targets) {
      const weight = (channel === 52 ? 1 : weights?.[channel] ?? 0) * scale;
      if (!weight) continue;
      for (let i = 0; i < points.length; i += 4) {
        const vertex = points[i] * 3;
        deltas[vertex] += weight * points[i + 1];
        deltas[vertex + 1] += weight * points[i + 2];
        deltas[vertex + 2] += weight * points[i + 3];
      }
    }
    return deltas;
  });
}

import type { AvatarMesh, ContactRig, Mapping } from './types';

export const POINTS = 478;
export const CHANNELS = 52;
export const INNER_UP = [78,191,80,81,82,13,312,311,310,415,308];
export const INNER_DOWN = [78,95,88,178,87,14,317,402,318,324,308];
const OUTER_UP = [61,185,40,39,37,0,267,269,270,409,291];
const OUTER_DOWN = [61,146,91,181,84,17,314,405,321,375,291];

export function parseMapping(data: Record<string, unknown>): Mapping {
  const array = (key: string, count?: number) => {
    const value = data[key];
    if (!Array.isArray(value) || (count !== undefined && value.length !== count) || !value.every(Number.isFinite)) {
      throw new Error(`Invalid face mapping: ${key}.`);
    }
    return Float32Array.from(value);
  };
  if (data.kind !== 'linear' && data.kind !== 'local') throw new Error('Unknown mapping type.');
  const local = data.kind === 'local';
  const featureCount = local ? Number(data.featureCount) : 0;
  const exampleCount = local ? Number(data.exampleCount) : 0;
  if (local && (!Number.isInteger(featureCount) || featureCount <= 0 || !Number.isInteger(exampleCount) || exampleCount <= 0 || !(Number(data.bandwidth) > 0))) {
    throw new Error('Invalid local mapping dimensions.');
  }
  return {
    kind: data.kind, reference: array('reference', POINTS * 2), basis: array('basis', POINTS * 2 * CHANNELS),
    sourceReference: array('sourceReference', CHANNELS), featureCount, exampleCount,
    featureBasis: local ? array('featureBasis', featureCount * CHANNELS) : new Float32Array(),
    training: local ? array('training', featureCount * exampleCount) : new Float32Array(),
    solution: local ? array('solution', exampleCount * POINTS * 2) : new Float32Array(),
    bandwidth: local ? Number(data.bandwidth) : 1
  };
}

export function sampleTrack(frames: Float32Array, time: number, result: Float32Array) {
  const count = frames.length / CHANNELS;
  if (!count) throw new Error('No shape frames are available.');
  const position = Math.max(0, Math.min(count - 1, time * 60));
  const left = Math.floor(position); const right = Math.min(count - 1, left + 1); const t = position - left;
  for (let c = 0; c < CHANNELS; c++) result[c] = frames[left * CHANNELS + c] * (1 - t) + frames[right * CHANNELS + c] * t;
}

export class ShapeSampler {
  readonly local = new Float32Array(POINTS * 2);
  readonly image = new Float32Array((POINTS + 36) * 2);
  private delta = new Float32Array(CHANNELS);
  private gaps = new Float32Array(9);
  private features: Float32Array;
  private zeroKernels: Float64Array;
  readonly mesh: AvatarMesh; readonly map: Mapping; readonly contact: ContactRig;
  constructor(mesh: AvatarMesh, map: Mapping, contact: ContactRig) {
    this.mesh=mesh;this.map=map;this.contact=contact;
    this.features = new Float32Array(map.featureCount);
    this.zeroKernels = new Float64Array(map.exampleCount);
    for (let e = 0; e < map.exampleCount; e++) {
      let squared = 0;
      for (let f = 0; f < map.featureCount; f++) squared += map.training[e * map.featureCount + f] ** 2;
      this.zeroKernels[e] = Math.exp(-squared / (2 * map.bandwidth ** 2));
    }
  }
  sample(weights: Float32Array, strength: number, correctContact: boolean) {
    const { map, mesh, local, delta } = this;
    for (let c = 0; c < CHANNELS; c++) delta[c] = weights[c] - map.sourceReference[c];
    for (let j = 0; j < local.length; j++) {
      let value = map.reference[j]; const offset = j * CHANNELS;
      for (let c = 0; c < CHANNELS; c++) value += map.basis[offset + c] * delta[c];
      local[j] = value;
    }
    if (map.kind === 'local') {
      let zeroDistance = 0;
      for (let f = 0; f < map.featureCount; f++) {
        let value = 0;
        for (let c = 0; c < CHANNELS; c++) value += map.featureBasis[f * CHANNELS + c] * delta[c];
        this.features[f] = value; zeroDistance += value * value;
      }
      const zero = Math.exp(-zeroDistance / (2 * map.bandwidth ** 2));
      for (let e = 0; e < map.exampleCount; e++) {
        let distance = 0;
        for (let f = 0; f < map.featureCount; f++) distance += (this.features[f] - map.training[e * map.featureCount + f]) ** 2;
        const k = Math.exp(-distance / (2 * map.bandwidth ** 2)) - zero * this.zeroKernels[e];
        for (let j = 0; j < local.length; j++) local[j] += k * map.solution[e * local.length + j];
      }
    }
    if (correctContact) this.applyContact(weights);
    const { center, width, rotation: r } = mesh.frame;
    for (let i = 0; i < POINTS; i++) {
      const x = map.reference[2 * i] + (local[2 * i] - map.reference[2 * i]) * strength;
      const y = map.reference[2 * i + 1] + (local[2 * i + 1] - map.reference[2 * i + 1]) * strength;
      this.image[2 * i] = (x * r[0] + y * r[1]) * width + center[0];
      this.image[2 * i + 1] = (x * r[2] + y * r[3]) * width + center[1];
    }
    this.image.set(mesh.ring, POINTS * 2);
    return this.image;
  }
  private applyContact(weights: Float32Array) {
    const { contact, map, local, mesh } = this;
    const gaps = this.gaps;
    for (let p = 0; p < 9; p++) {
      let gap = contact.rest[p];
      for (let c = 0; c < CHANNELS; c++) gap += contact.basis[p * CHANNELS + c] * weights[c];
      gaps[p] = gap;
    }
    gaps.sort((a, b) => a - b);
    let movement = 0;
    // Upper-face controls do not determine contact strength.
    for (let c = 14; c <= 40; c++) movement += (weights[c] - map.sourceReference[c]) ** 2;
    movement += (weights[46] - map.sourceReference[46]) ** 2;
    const amount = gaps[4] <= 0.1 ? Math.min(1, Math.sqrt(movement) / 0.15) : 0;
    for (let p = 0; p < 9; p++) {
      const upper = INNER_UP[p + 1]; const lower = INNER_DOWN[p + 1];
      const gap = local[lower * 2 + 1] - local[upper * 2 + 1];
      const wanted = Math.max(0.001, Math.min(0.008, mesh.contactGap[p]));
      const shift = Math.max(-0.15, Math.min(0.15, (wanted - gap) * amount / 2));
      local[upper * 2 + 1] -= shift; local[lower * 2 + 1] += shift;
      local[OUTER_UP[p + 1] * 2 + 1] -= shift; local[OUTER_DOWN[p + 1] * 2 + 1] += shift;
    }
  }
}

export type Sample = { id: string; label: string; text: string; url: string; duration: number };
export type AvatarSummary = { id: string; name: string; url: string; thumbnail: string };
export type ContactRig = { rest: number[]; basis: number[] };
export type Catalog = {
  avatars: AvatarSummary[]; samples: Sample[];
  model: { url: string; sha256: string; bytes: number }; contact: ContactRig;
};
export type Crop = { x: number; y: number; w: number; h: number };
export type AvatarMesh = {
  id: string; name: string; image: string; imageSize: [number, number];
  sourceUV: number[]; ring: number[]; triangles: number[]; crop: Crop;
  frame: { center: number[]; width: number; rotation: number[] };
  cheek?: { basis: number[]; blend: number[]; method: string };
  fitHash: string; calibration: string; contactGap: number[];
};
export type Mapping = {
  kind: 'linear' | 'local'; reference: Float32Array; basis: Float32Array; sourceReference: Float32Array;
  featureBasis: Float32Array; featureCount: number; training: Float32Array; exampleCount: number;
  solution: Float32Array; bandwidth: number;
};
export type Track = { frames: Float32Array; count: number; inferenceMs: number; duration: number };
export type Runtime = 'webgpu' | 'cpu';

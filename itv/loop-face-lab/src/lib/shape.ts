import { LipContactConstraint } from './contact.ts';
import type { AvatarMesh, ContactRig, Mapping } from './types';

export const POINTS = 478;
export const CHANNELS = 52;
export { INNER_UP, INNER_DOWN } from './contact.ts';

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
  private lipContact: LipContactConstraint;
  private beforeCheeks = new Float32Array(POINTS*2);
  private cheekTriangles: number[]=[];
  cheekGain=1;
  private cheekBasis: Float32Array;
  private cheekBlend: Float32Array;
  private delta = new Float32Array(CHANNELS);
  private gaps = new Float32Array(9);
  private features: Float32Array;
  private zeroKernels: Float64Array;
  readonly mesh: AvatarMesh; readonly map: Mapping; readonly contact: ContactRig;
  constructor(mesh: AvatarMesh, map: Mapping, contact: ContactRig) {
    this.mesh=mesh;this.map=map;this.contact=contact;
    this.lipContact=new LipContactConstraint(mesh,map.reference);
    this.cheekBasis=Float32Array.from(mesh.cheek?.basis??[]);
    this.cheekBlend=Float32Array.from(mesh.cheek?.blend??[]);
    if(mesh.cheek && (this.cheekBasis.length!==POINTS*2*CHANNELS || this.cheekBlend.length!==POINTS || !this.cheekBasis.every(Number.isFinite) || !this.cheekBlend.every(v=>Number.isFinite(v)&&v>=0&&v<=1)))throw new Error("Invalid cheek transfer data.");
    for(let t=0;t<mesh.triangles.length;t+=3){
      const a=mesh.triangles[t],b=mesh.triangles[t+1],c=mesh.triangles[t+2];
      if(a<POINTS&&b<POINTS&&c<POINTS&&(this.cheekBlend[a]>0||this.cheekBlend[b]>0||this.cheekBlend[c]>0))this.cheekTriangles.push(a,b,c);
    }
    this.features = new Float32Array(map.featureCount);
    this.zeroKernels = new Float64Array(map.exampleCount);
    for (let e = 0; e < map.exampleCount; e++) {
      let squared = 0;
      for (let f = 0; f < map.featureCount; f++) squared += map.training[e * map.featureCount + f] ** 2;
      this.zeroKernels[e] = Math.exp(-squared / (2 * map.bandwidth ** 2));
    }
  }
  sample(weights: Float32Array, strength: number, correctContact: boolean, cheekStrength = 0) {
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
    this.cheekGain=1;
    if(cheekStrength>0 && this.cheekBasis.length){
      this.beforeCheeks.set(local);
      for(let i=0;i<POINTS;i++){
        const amount=this.cheekBlend[i]*Math.max(0,Math.min(1,cheekStrength));
        if(!amount)continue;
        for(let axis=0;axis<2;axis++){
          const j=i*2+axis;let motion=0;
          for(let c=0;c<CHANNELS;c++)motion+=this.cheekBasis[j*CHANNELS+c]*delta[c];
          local[j]+=(map.reference[j]+motion-local[j])*amount;
        }
      }
      this.limitCheekDeformation();
    }
    for(let j=0;j<local.length;j++)local[j]=map.reference[j]+(local[j]-map.reference[j])*strength;
    if(correctContact && strength>0)this.applyContact(weights);
    const { center, width, rotation: r } = mesh.frame;
    for (let i = 0; i < POINTS; i++) {
      const x = local[2 * i];
      const y = local[2 * i + 1];
      this.image[2 * i] = (x * r[0] + y * r[1]) * width + center[0];
      this.image[2 * i + 1] = (x * r[2] + y * r[3]) * width + center[1];
    }
    this.image.set(mesh.ring, POINTS * 2);
    return this.image;
  }
  private limitCheekDeformation(){
    const base=this.beforeCheeks;const next=this.local;
    // Local clipping prevents one poor triangle from freezing both cheeks.
    // A final global pass handles any remaining coupled constraints.
    for(let pass=0;pass<13;pass++){
    let gain=1,changed=false;
    for(let t=0;t<this.cheekTriangles.length;t+=3){
      const a=this.cheekTriangles[t]*2,b=this.cheekTriangles[t+1]*2,c=this.cheekTriangles[t+2]*2;
      const ux=base[b]-base[a],uy=base[b+1]-base[a+1],vx=base[c]-base[a],vy=base[c+1]-base[a+1];
      const dux=(next[b]-base[b])-(next[a]-base[a]),duy=(next[b+1]-base[b+1])-(next[a+1]-base[a+1]);
      const dvx=(next[c]-base[c])-(next[a]-base[a]),dvy=(next[c+1]-base[c+1])-(next[a+1]-base[a+1]);
      const area=ux*vy-uy*vx;if(Math.abs(area)<1e-10)continue;
      // Area is quadratic along the deformation. Stop before its first root
      // at 10% of the original area, including a small numerical margin.
      const sign=Math.sign(area),constant=Math.abs(area)*.9;
      const linear=(dux*vy+ux*dvy-duy*vx-uy*dvx)*sign;
      const quadratic=(dux*dvy-duy*dvx)*sign;
      let root=Infinity;
      if(Math.abs(quadratic)<1e-14){if(linear<0)root=-constant/linear;}
      else{
        const discriminant=linear*linear-4*quadratic*constant;
        if(discriminant>=0){
          const sqrt=Math.sqrt(discriminant);const r1=(-linear-sqrt)/(2*quadratic),r2=(-linear+sqrt)/(2*quadratic);
          if(r1>0)root=Math.min(root,r1);if(r2>0)root=Math.min(root,r2);
        }
      }
      if(root<=1){
        const factor=root*.95;gain=Math.min(gain,factor);changed=true;
        if(pass<12)for(let corner=0;corner<3;corner++){
          const index=this.cheekTriangles[t+corner]*2;
          next[index]=base[index]+(next[index]-base[index])*factor;
          next[index+1]=base[index+1]+(next[index+1]-base[index+1])*factor;
        }
      }
    }
    this.cheekGain=Math.min(this.cheekGain,gain);
    if(!changed)break;
    if(pass===12&&gain<1)for(let j=0;j<next.length;j++)next[j]=base[j]+(next[j]-base[j])*gain;
    }
  }
  private applyContact(weights: Float32Array) {
    const { contact, map, local } = this;
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
    const full=contact.capture?.full??.1,release=contact.capture?.release??full;
    let capture=gaps[4]<=full?1:0;
    if(gaps[4]>full&&gaps[4]<release){const t=(release-gaps[4])/(release-full);capture=t*t*(3-2*t);}
    const amount=capture*Math.min(1,Math.sqrt(movement)/.15);
    this.lipContact.apply(local,amount);
  }
}

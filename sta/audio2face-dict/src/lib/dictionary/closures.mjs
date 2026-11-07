import { CHANNELS, FPS, SAMPLE_RATE } from './index.mjs';
import { warpSourceTime } from './retime-closures.mjs';
import advanceModel from '../../../data/advance-closure-model.json' with { type: 'json' };

export const CONTACT_GAP = 0.10;
export const closurePhone = (phone) => phone === 'M' || phone === 'P' || phone === 'B';

const acousticOffsets = [-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06];
const halfWindow = 0.010;
const featureCount = 6 + 2 * acousticOffsets.length;

export function validClosureModel(model) {
  return model?.version === 1 && Number.isFinite(model.bias) &&
    [model.mean, model.scale, model.weights].every((values) =>
      Array.isArray(values) && values.length === featureCount && values.every(Number.isFinite)) &&
    model.scale.every((value) => value > 0);
}

export function closureFeatures(pcm, phone, time) {
  const duration = phone.end - phone.start;
  const phase = (time - phone.start) / duration - 0.5;
  const features = [phase, phase * phase, Math.log(duration), phone.phone === 'M' ? 1 : 0, phone.phone === 'B' ? 1 : 0];
  const energy = [];
  const high = [];
  for (const offset of acousticOffsets) {
    const start = Math.max(1, Math.round((time + offset - halfWindow) * SAMPLE_RATE));
    const end = Math.min(pcm.length, Math.round((time + offset + halfWindow) * SAMPLE_RATE));
    let rms = 0, difference = 0;
    for (let i = start; i < end; i++) {
      rms += pcm[i] ** 2;
      difference += (pcm[i] - pcm[i - 1]) ** 2;
    }
    const count = Math.max(1, end - start);
    energy.push(Math.log(1e-6 + rms / count));
    high.push(Math.log((1e-6 + difference / count) / (1e-6 + rms / count)));
  }
  const peak = Math.max(...energy);
  return [...features, ...energy.map((value) => value - peak), ...high, peak];
}

export function closurePrediction(model, features) {
  let value = model.bias;
  for (let i = 0; i < features.length; i++) {
    value += model.weights[i] * (features[i] - model.mean[i]) / model.scale[i];
  }
  return 1 / (1 + Math.exp(-value));
}

export function advanceClosureFeatures(pcm, phone, time) {
  const features = closureFeatures(pcm, phone, time);
  // Alignment intervals are often only 20 ms long. Use an absolute timing
  // scale for anticipation, rather than stretching it by phone duration.
  features[0] = (time - phone.start) / .1;
  features[1] = features[0] ** 2;
  return features;
}

export function makeClosureGuide(dictionary, phones, pcm, advance = false) {
  if (advance && !dictionary.closureModel) return;
  const model = advance ? advanceModel : dictionary.closureModel;
  if (!model || !dictionary.lipRig) return;
  if (advance && !validClosureModel(model)) throw new Error('Invalid advance closure model');
  const samples = new Map();
  const probability = new Float32Array(Math.ceil(pcm.length * FPS / SAMPLE_RATE));
  probability.fill(-1);
  for (const phone of phones) {
    if (!closurePhone(phone.phone)) continue;
    const band = advance ? model.timingProfile.phones[phone.phone] : undefined;
    const from = Math.max(0, Math.ceil((phone.start + (band ? band.onsetRangeMs[0] / 1000 : -0.02)) * FPS));
    const to = Math.min(probability.length, Math.ceil((phone.end + (band ? band.releaseRangeMs[1] / 1000 : 0.02)) * FPS));
    const points = [];
    for (let frame = from; frame < to; frame++) {
      const time = frame / FPS;
      const confidence = closurePrediction(model, advance ? advanceClosureFeatures(pcm, phone, time) : closureFeatures(pcm, phone, time));
      points.push({ time, confidence });
    }
    const peak = Math.max(0, ...points.map(({ confidence }) => confidence));
    const canFavorOpen = peak < 0.25;
    for (const point of points) {
      // Inconclusive frames should not erase contact in a phone that has
      // credible closure evidence somewhere in its window.
      if (!canFavorOpen && point.confidence < 0.25) point.confidence = 0.25;
      const frame = Math.round(point.time * FPS);
      probability[frame] = Math.max(probability[frame], point.confidence);
    }
    samples.set(phone, points);
  }

  const { neutral, deltas } = dictionary.lipRig;
  const center = Math.floor(neutral.length / 2);
  const basis = deltas.flatMap((values, index) =>
    values[center] ? [{ index, delta: values[center] }] : []);
  function gapAt(clip, time) {
    const position = Math.max(0, Math.min(clip.frameCount - 1, time * FPS));
    const frame = Math.floor(position);
    const from = (clip.frameOffset + frame) * CHANNELS;
    const to = (clip.frameOffset + Math.min(clip.frameCount - 1, frame + 1)) * CHANNELS;
    let gap = neutral[center];
    for (const { index, delta } of basis) {
      const a = dictionary.weights[from + index];
      const b = dictionary.weights[to + index];
      gap += delta * (a + (b - a) * (position - frame));
    }
    return gap;
  }

  const frameGap = (weights, offset) => basis.reduce((gap, { index, delta }) =>
    gap + delta * weights[offset + index], neutral[center]);
  return { samples, probability, gapAt, frameGap, phones, advance, profile: advance ? model.timingProfile : undefined };
}

export function closureCandidateCost(guide, clip, source, target, warps) {
  const targetKnots = target.map((phone) => (phone.start + phone.end) / 2);
  const sourceKnots = source.map((phone) => (phone.start + phone.end) / 2);
  if (guide.advance) {
    let cost = 0, frames = 0, left = 0;
    for (let frame = Math.ceil(targetKnots[0] * FPS); frame / FPS < targetKnots.at(-1); frame++) {
      const confidence = guide.probability[frame];
      if (!(confidence >= 0)) continue;
      frames++;
      if (confidence > .1 && confidence < .85) continue;
      const time = frame / FPS;
      while (left + 2 < targetKnots.length && time > targetKnots[left + 1]) left++;
      const phase = (time - targetKnots[left]) / (targetKnots[left + 1] - targetKnots[left]);
      const mapped = sourceKnots[left] + phase * (sourceKnots[left + 1] - sourceKnots[left]);
      const gap = guide.gapAt(clip, warpSourceTime(warps, mapped));
      const closed = Math.min(1, Math.max(0, (.25 - gap) / .15));
      cost += confidence >= .85 ? (confidence - .85) / .15 * (1 - closed) : (.1 - confidence) / .1 * closed;
    }
    return frames ? cost / frames : 0;
  }
  let cost = 0;
  for (const [i, phone] of target.entries()) {
    const points = guide.samples.get(phone);
    if (!points?.length) continue;
    let difference = 0;
    for (const { time, confidence } of points) {
      if (confidence >= 0.12 && confidence <= 0.70) continue;
      let left = time <= targetKnots[i] ? i - 1 : i;
      left = Math.max(0, Math.min(target.length - 2, left));
      const phase = (time - targetKnots[left]) / (targetKnots[left + 1] - targetKnots[left]);
      // Compare only the part of this phoneme that this clip contributes.
      if (phase < 0 || phase > 1) continue;
      const gap = guide.gapAt(clip, warpSourceTime(warps, sourceKnots[left] + phase * (sourceKnots[left + 1] - sourceKnots[left])));
      const closed = Math.min(1, Math.max(0, (0.25 - gap) / 0.15));
      if (confidence > 0.70) difference += (confidence - 0.70) * (1 - closed);
      else difference += (0.12 - confidence) * closed;
    }
    cost += difference / points.length;
  }
  return cost;
}

export function closureBlend(guide, time, left, leftTime, right, rightTime, amount) {
  if (!guide || amount <= 0 || amount >= 1) return amount;
  const confidence = guide.probability[Math.round(time * FPS)];
  if (guide.advance) {
    if (!(confidence >= 0) || (confidence > .1 && confidence < .85)) return amount;
    const leftGap = guide.gapAt(left, leftTime), rightGap = guide.gapAt(right, rightTime);
    const change = rightGap - leftGap;
    if (Math.abs(change) < .025) return amount;
    // Closing is allowed only if a donor already contains actual contact.
    if (confidence >= .85 && Math.min(leftGap, rightGap) > CONTACT_GAP) return amount;
    const blended = leftGap + change * amount;
    const target = confidence >= .85 ? .085 : .14;
    if ((confidence >= .85 && blended <= target) || (confidence <= .1 && blended >= target)) return amount;
    const correction = (target - blended) / change;
    const weight = confidence >= .85 ? (confidence - .85) / .15 : (.1 - confidence) / .1;
    const limit = .65 * 4 * amount * (1 - amount) * weight;
    return Math.min(1, Math.max(0, amount + Math.min(limit, Math.max(-limit, correction))));
  }
  if (!(confidence >= 0) || (confidence >= 0.12 && confidence <= 0.65)) return amount;
  const leftGap = guide.gapAt(left, leftTime);
  const rightGap = guide.gapAt(right, rightTime);
  const change = rightGap - leftGap;
  if (Math.abs(change) < 0.025) return amount;
  const blended = leftGap + change * amount;
  const target = confidence > 0.65 ? 0.085 : 0.14;
  const correction = (target - blended) / change;
  if ((confidence > 0.65 && blended <= target) || (confidence < 0.12 && blended >= target)) return amount;
  const confidenceWeight = confidence > 0.65 ? (confidence - 0.65) / 0.35 : (0.12 - confidence) / 0.12;
  const limit = 0.65 * 4 * amount * (1 - amount) * confidenceWeight;
  return Math.min(1, Math.max(0, amount + Math.min(limit, Math.max(-limit, correction))));
}

import { CHANNELS, FPS } from './index.mjs';

export function validSelectionModel(model) {
  if (model?.version !== 1 || !Array.isArray(model.labels) || !model.labels.length ||
      !model.labels.every((phone) => /^[A-Z]+$/.test(phone)) || new Set(model.labels).size !== model.labels.length) return false;
  const size = 9 + 3 * model.labels.length;
  return [model.mean, model.scale].every((row) => Array.isArray(row) && row.length === size && row.every(Number.isFinite)) &&
    model.scale.every((value) => value > 0) && Array.isArray(model.bias) && model.bias.length === 3 && model.bias.every(Number.isFinite) &&
    Array.isArray(model.weights) && model.weights.length === size && model.weights.every((row) => Array.isArray(row) && row.length === 3 && row.every(Number.isFinite));
}

export function selectionFeatures(phones, at, acoustics, labels) {
  const audio = acoustics[at];
  const values = [...audio.slice(0, 6), audio[6] * Math.max(0, 2 * audio[7] - 1), audio[7], Math.log(phones[at].end - phones[at].start)];
  for (const position of [at - 1, at, at + 1]) {
    for (const phone of labels) values.push(Number(phones[position]?.phone === phone));
  }
  return values;
}

export function predictLipTargets(model, phones, acoustics) {
  return phones.map((_, at) => {
    const values = selectionFeatures(phones, at, acoustics, model.labels).map((value, i) =>
      (value - model.mean[i]) / model.scale[i]);
    return model.bias.map((bias, j) => values.reduce((sum, value, i) => sum + value * model.weights[i][j], bias));
  });
}

export function predictedLipCost(dictionary, clip, source, target) {
  let cost = 0;
  const landmarks = [2, 4, 6];
  for (const [i, phone] of source.entries()) {
    const time = (phone.start + phone.end) / 2 * FPS;
    const frame = Math.max(0, Math.min(clip.frameCount - 1, time));
    const from = (clip.frameOffset + Math.floor(frame)) * CHANNELS;
    const to = (clip.frameOffset + Math.min(clip.frameCount - 1, Math.floor(frame) + 1)) * CHANNELS;
    for (const [j, landmark] of landmarks.entries()) {
      let gap = dictionary.lipRig.neutral[landmark];
      for (let c = 0; c < CHANNELS; c++) {
        const a = dictionary.weights[from + c], b = dictionary.weights[to + c];
        gap += dictionary.lipRig.deltas[c][landmark] * (a + (b - a) * (frame % 1));
      }
      cost += Math.min(4, ((gap - target[i][j]) / 0.6) ** 2) / (source.length * 3);
    }
  }
  return cost;
}

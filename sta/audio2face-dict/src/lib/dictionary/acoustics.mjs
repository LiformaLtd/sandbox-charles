import { SAMPLE_RATE } from './index.mjs';

// Small, gain-normalized descriptors; source PCM is never shipped to the browser.
export function describeSpeech(pcm, phones) {
  const rows = phones.map((phone) => {
    const energy = [], brightness = [];
    for (const phase of [0.2, 0.5, 0.8]) {
      const center = Math.round((phone.start + phase * (phone.end - phone.start)) * SAMPLE_RATE);
      let square = 0, difference = 0, count = 0;
      for (let i = Math.max(1, center - 160); i < Math.min(pcm.length, center + 160); i++) {
        square += pcm[i] ** 2;
        difference += (pcm[i] - pcm[i - 1]) ** 2;
        count++;
      }
      energy.push(Math.log(1e-7 + square / Math.max(1, count)));
      brightness.push(Math.log((1e-7 + difference) / (1e-7 + square)));
    }
    // Autocorrelation at 4 kHz, over 40 ms. Average before decimation.
    const window = new Float32Array(160);
    const start = Math.round((phone.start + phone.end) / 2 * SAMPLE_RATE) - 320;
    for (let i = 0; i < window.length; i++) {
      for (let j = 0; j < 4; j++) window[i] += (pcm[start + i * 4 + j] ?? 0) / 4;
    }
    const mean = window.reduce((sum, value) => sum + value, 0) / window.length;
    for (let i = 0; i < window.length; i++) window[i] -= mean;
    let strength = 0, lagAt = 0;
    for (let lag = 10; lag <= 50; lag++) {
      let product = 0, left = 0, right = 0;
      for (let i = lag; i < window.length; i++) {
        product += window[i] * window[i - lag];
        left += window[i] ** 2;
        right += window[i - lag] ** 2;
      }
      const correlation = product / Math.sqrt(1e-12 + left * right);
      if (correlation > strength) { strength = correlation; lagAt = lag; }
    }
    return [...energy, ...brightness, lagAt ? Math.log(4000 / lagAt) : 0, strength];
  });
  const energyMean = rows.reduce((sum, row) => sum + row[1], 0) / Math.max(1, rows.length);
  const voiced = rows.filter((row) => row[7] >= 0.7 && row[1] > energyMean - 3);
  const pitchMean = voiced.reduce((sum, row) => sum + row[6], 0) / Math.max(1, voiced.length);
  return rows.map((row) => row.map((value, i) =>
    Math.round(1000 * (i < 3 ? value - energyMean : i === 6 ? value - pitchMean : value)) / 1000));
}

export function acousticDistance(source, target) {
  if (!source || !target) return 0;
  let distance = 0;
  for (let i = 0; i < 3; i++) {
    distance += Math.min(4, ((source[i] - target[i]) / 2) ** 2) / 6;
    distance += Math.min(4, ((source[i + 3] - target[i + 3]) / 2) ** 2) / 6;
  }
  if (source[7] > 0.7 && target[7] > 0.7) distance += Math.min(2, ((source[6] - target[6]) / 0.35) ** 2) / 4;
  distance += (source[7] - target[7]) ** 2 / 4;
  return distance;
}

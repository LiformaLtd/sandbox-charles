import {
  F_MAX,
  F_MIN,
  HOP_LENGTH,
  N_FFT,
  N_MELS,
  SAMPLE_RATE,
  WIN_LENGTH,
} from './config';
import { createRadix2FFT, hannWindow } from './fft';

export interface AudioFeatures {
  mel: Float32Array;
  frames: number;
  nMels: number;
  duration: number;
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildMelFilterbank() {
  const nFreqs = N_FFT / 2 + 1;
  const melMin = hzToMel(F_MIN);
  const melMax = hzToMel(F_MAX);

  const melPoints = new Float32Array(N_MELS + 2);
  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] = melMin + (i / (N_MELS + 1)) * (melMax - melMin);
  }

  const hzPoints = Array.from(melPoints, (m) => melToHz(m));
  const bins = hzPoints.map((hz) => Math.floor(((N_FFT + 1) * hz) / SAMPLE_RATE));

  const filters: { start: number; weights: Float32Array }[] = [];
  for (let m = 1; m <= N_MELS; m++) {
    const f = new Float32Array(nFreqs);
    const left = bins[m - 1];
    const center = bins[m];
    const right = bins[m + 1];

    for (let k = left; k < center; k++) {
      if (k >= 0 && k < nFreqs && center > left) {
        f[k] = (k - left) / (center - left);
      }
    }
    for (let k = center; k < right; k++) {
      if (k >= 0 && k < nFreqs && right > center) {
        f[k] = (right - k) / (right - center);
      }
    }
    const start = f.findIndex((weight) => weight !== 0);
    let end = nFreqs;
    while (end > 0 && f[end - 1] === 0) end--;
    filters.push(start < 0
      ? { start: 0, weights: new Float32Array(0) }
      : { start, weights: f.slice(start, end) });
  }

  return filters;
}

const MEL_FILTERS = buildMelFilterbank();
const WINDOW = hannWindow(WIN_LENGTH);
const fft = createRadix2FFT(N_FFT);

function reflectPad(signal: Float32Array, pad: number): Float32Array {
  const out = new Float32Array(signal.length + pad * 2);
  const n = signal.length;
  for (let i = 0; i < out.length; i++) {
    const src = i - pad;
    let idx = src;
    while (idx < 0 || idx >= n) {
      if (idx < 0) idx = -idx;
      if (idx >= n) idx = 2 * n - 2 - idx;
    }
    out[i] = signal[idx];
  }
  return out;
}

export async function resampleTo16k(samples: Float32Array, sampleRate: number): Promise<Float32Array> {
  if (samples.length < 2) throw new Error('Speech synthesis returned no usable audio.');
  const length = Math.round(samples.length * SAMPLE_RATE / sampleRate);
  const context = new OfflineAudioContext(1, length, SAMPLE_RATE);
  const source = context.createBufferSource();
  source.buffer = context.createBuffer(1, samples.length, sampleRate);
  source.buffer.getChannelData(0).set(samples);
  source.connect(context.destination);
  source.start();
  return (await context.startRendering()).getChannelData(0);
}

export function extractLogMel(signal16k: Float32Array): AudioFeatures {
  const padded = reflectPad(signal16k, N_FFT / 2);
  const nFrames = Math.max(1, Math.floor((padded.length - N_FFT) / HOP_LENGTH) + 1);
  const nFreqs = N_FFT / 2 + 1;

  const mel = new Float32Array(nFrames * N_MELS);
  const real = new Float32Array(N_FFT);
  const imag = new Float32Array(N_FFT);
  const power = new Float32Array(nFreqs);

  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP_LENGTH;

    real.fill(0, WIN_LENGTH);
    imag.fill(0);

    for (let i = 0; i < WIN_LENGTH; i++) {
      real[i] = padded[start + i] * WINDOW[i];
    }

    fft(real, imag);

    for (let k = 0; k < nFreqs; k++) {
      const re = real[k];
      const im = imag[k];
      power[k] = re * re + im * im;
    }

    for (let m = 0; m < N_MELS; m++) {
      const { start, weights } = MEL_FILTERS[m];
      let e = 0;
      for (let k = 0; k < weights.length; k++) {
        e += power[start + k] * weights[k];
      }
      mel[t * N_MELS + m] = Math.log(e + 1e-6);
    }
  }

  let sum = 0;
  for (let i = 0; i < mel.length; i++) sum += mel[i];
  const mean = sum / mel.length;

  let varSum = 0;
  for (let i = 0; i < mel.length; i++) {
    const d = mel[i] - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / mel.length);
  const denom = std + 1e-5;

  for (let i = 0; i < mel.length; i++) {
    mel[i] = (mel[i] - mean) / denom;
  }

  return {
    mel,
    frames: nFrames,
    nMels: N_MELS,
    duration: signal16k.length / SAMPLE_RATE,
  };
}

const fftSize = 512;
const frames = 101;
const bins = fftSize / 2 + 1;
const reverse = Uint16Array.from({ length: fftSize }, (_, index) => {
  let result = 0;
  for (let bit = 0; bit < 9; bit++) {
    result = (result << 1) | (index & 1);
    index >>= 1;
  }
  return result;
});
const cosine = Float64Array.from({ length: fftSize / 2 }, (_, i) => Math.cos(2 * Math.PI * i / fftSize));
const sine = Float64Array.from({ length: fftSize / 2 }, (_, i) => -Math.sin(2 * Math.PI * i / fftSize));

export function createLogMel(window: Float32Array, mel: Float32Array) {
  const bands = Array.from({ length: 64 }, (_, band) => {
    const weights = mel.subarray(band * bins, (band + 1) * bins);
    const start = weights.findIndex((weight) => weight > 0);
    const end = weights.findLastIndex((weight) => weight > 0) + 1;
    return { start, weights: weights.slice(start, end) };
  });
  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);
  const power = new Float64Array(bins);
  const output = new Float32Array(bands.length * frames);
  return (pcm: Float32Array) => {
    for (let frame = 0; frame < frames; frame++) {
      const offset = frame * 160 - 200;
      real.fill(0);
      imag.fill(0);
      for (let i = Math.max(0, -offset); i < Math.min(window.length, pcm.length - offset); i++) {
        real[reverse[i]] = pcm[offset + i] * window[i];
      }
      for (let size = 2; size <= fftSize; size *= 2) {
        const half = size / 2;
        const stride = fftSize / size;
        for (let i = 0; i < fftSize; i += size) {
          for (let j = 0; j < half; j++) {
            const a = i + j;
            const b = a + half;
            const k = j * stride;
            const r = real[b] * cosine[k] - imag[b] * sine[k];
            const q = real[b] * sine[k] + imag[b] * cosine[k];
            real[b] = real[a] - r;
            imag[b] = imag[a] - q;
            real[a] += r;
            imag[a] += q;
          }
        }
      }
      for (let i = 0; i < bins; i++) power[i] = real[i] ** 2 + imag[i] ** 2;
      for (let band = 0; band < bands.length; band++) {
        const { start, weights } = bands[band];
        let sum = 0;
        for (let i = 0; i < weights.length; i++) sum += power[start + i] * weights[i];
        output[band * frames + frame] = Math.log1p(sum * 10);
      }
    }
    return output;
  };
}

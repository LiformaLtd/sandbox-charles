// Keep the filter history and sample phase across worker messages so
// changing the size of a speech chunk cannot introduce an audio seam.
const filterCache = new Map();

function makeFilters(phases, radius, cutoff) {
  if (!filterCache.has(phases)) {
    const filters = Array.from({ length: phases }, (_, phase) => {
      const weights = Array.from({ length: 2 * radius + 1 }, (_, i) => {
        const x = i - radius - phase / phases;
        const angle = Math.PI * x * cutoff;
        const sinc = angle ? Math.sin(angle) / angle : 1;
        return Math.abs(x) <= radius ? sinc * (0.5 + 0.5 * Math.cos(Math.PI * x / radius)) : 0;
      });
      const total = weights.reduce((sum, weight) => sum + weight, 0);
      return weights.map((weight) => weight / total);
    });
    filterCache.set(phases, filters);
  }
  return filterCache.get(phases);
}

export async function* to16k(chunks, rate) {
  if (rate === 16000) {
    yield* chunks;
    return;
  }
  if (rate !== 24000 && rate !== 44100) throw new Error(`Unsupported speech sample rate: ${rate} Hz.`);

  // Precompute one filter per fractional input position in the rational ratio.
  const phases = rate === 24000 ? 2 : 160;
  const stride = rate * phases / 16000;
  const radius = rate === 24000 ? 12 : 24;
  const filters = makeFilters(phases, radius, 0.93 * 16000 / rate);

  let buffer = new Float32Array(0);
  let base = 0;
  let received = 0;
  let next = 0;
  function drain(end = false) {
    const samples = [];
    const limit = Math.round(received * phases / stride);
    while (next < limit) {
      const center = Math.floor(next * stride / phases);
      if (!end && center + radius >= received) break;
      const weights = filters[(next * stride) % phases];
      let value = 0;
      for (let i = 0; i < weights.length; i++) {
        value += (buffer[center - radius + i - base] ?? 0) * weights[i];
      }
      samples.push(Math.max(-1, Math.min(1, value)));
      next++;
    }
    const keep = Math.max(base, Math.floor(next * stride / phases) - radius);
    buffer = buffer.slice(keep - base);
    base = keep;
    return Float32Array.from(samples);
  }

  for await (const chunk of chunks) {
    const joined = new Float32Array(buffer.length + chunk.length);
    joined.set(buffer);
    joined.set(chunk, buffer.length);
    buffer = joined;
    received += chunk.length;
    const samples = drain();
    if (samples.length) yield samples;
  }
  const final = drain(true);
  if (final.length) yield final;
}

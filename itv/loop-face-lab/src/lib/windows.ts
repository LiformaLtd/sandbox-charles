export function windowFrames(start: number, sampleCount: number) {
  const frames: { local: number; global: number }[] = [];
  for (let i = 15; i < 45; i++) {
    const sample = start + Math.floor(i * 16000 / 60);
    if (sample >= 0 && sample < sampleCount) frames.push({ local: i, global: Math.ceil(sample * 60 / 16000) });
  }
  return frames;
}

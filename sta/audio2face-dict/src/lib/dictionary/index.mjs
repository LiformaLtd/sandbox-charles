export const SAMPLE_RATE = 16_000;
export const FPS = 60;
export const CHANNELS = 52;

export function speechRuns(phones, pcm, silenceRms = 0.008) {
  const runs = [];
  let start = 0;
  for (let i = 1; i < phones.length; i++) {
    const from = Math.round(phones[i - 1].end * SAMPLE_RATE);
    const to = Math.round(phones[i].start * SAMPLE_RATE);
    let quiet = 0;
    for (let at = from; at + 160 <= to; at += 160) {
      let energy = 0;
      for (let j = at; j < at + 160; j++) energy += pcm[j] ** 2;
      quiet = energy / 160 <= silenceRms ** 2 ? quiet + 160 : 0;
      if (quiet >= 1920) {
        runs.push({ from: start, to: i, pause: { start: (at + 160 - quiet) / SAMPLE_RATE, end: (at + 160) / SAMPLE_RATE } });
        start = i;
        break;
      }
    }
    // The quiet region may extend past the first 120 ms. Keep it contiguous.
    const pause = runs.at(-1)?.pause;
    if (start === i && pause) {
      for (let at = Math.round(pause.end * SAMPLE_RATE); at + 160 <= to; at += 160) {
        let energy = 0;
        for (let j = at; j < at + 160; j++) energy += pcm[j] ** 2;
        if (energy / 160 > silenceRms ** 2) break;
        pause.end = (at + 160) / SAMPLE_RATE;
      }
    }
  }
  if (phones.length) runs.push({ from: start, to: phones.length });
  return runs;
}

export const key = (phones, start, count) =>
  phones.slice(start, start + count).map(({ phone }) => phone).join(' ');

export function buildIndex(clips) {
  const index = Object.create(null);
  for (const [clipIndex, clip] of clips.entries()) {
    for (const { from, to } of clip.runs) {
      for (let start = from; start < to; start++) {
        for (let count = 1; count <= 3 && start + count <= to; count++) {
          const entry = index[key(clip.phones, start, count)] ??= [];
          entry.push({ clip: clipIndex, start, isolated: to - from === 1 });
        }
      }
    }
  }
  return index;
}

export function validatePhones(phones, duration) {
  let previousEnd = 0;
  for (const { phone, start, end } of phones) {
    if (!/^[A-Z]+$/.test(phone) || !Number.isFinite(start) || !Number.isFinite(end) ||
        start < previousEnd || end <= start || end > duration) {
      throw new Error('Invalid phoneme timing or label from TinyAligner.');
    }
    previousEnd = end;
  }
  if (!phones.length) throw new Error('TinyAligner returned no phonemes.');
}

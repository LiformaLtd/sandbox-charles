import { CHANNELS, FPS } from './index.mjs';

// Keep an existing contact if reranking loses it. This preserves a recorded
// baseline event, not a newly invented closing pose, and handles shared events once.
export function retainAdvanceContacts(frames, baseline, guide) {
  if (frames.length !== baseline.length) throw new Error('Baseline track duration differs');
  const ranges = guide.phones.flatMap((phone) => {
    const band = guide.profile.phones[phone.phone];
    return band ? [{ start: phone.start + band.onsetRangeMs[0] / 1000, end: phone.end + band.releaseRangeMs[1] / 1000 }] : [];
  });
  const count = frames.length / CHANNELS;
  let retained = 0;
  for (let first = 0; first < count; first++) {
    if (guide.frameGap(baseline, first * CHANNELS) > .1) continue;
    let last = first;
    while (last + 1 < count && guide.frameGap(baseline, (last + 1) * CHANNELS) <= .1) last++;
    const relevant = ranges.some(({ start, end }) => first / FPS <= end && last / FPS >= start);
    let survives = false;
    for (let frame = Math.max(0, first - 1); frame <= Math.min(count - 1, last + 1); frame++) {
      if (guide.frameGap(frames, frame * CHANNELS) <= .1) { survives = true; break; }
    }
    if (relevant && !survives) {
      for (let frame = Math.max(0, first - 3); frame <= Math.min(count - 1, last + 3); frame++) {
        const distance = Math.max(first - frame, frame - last, 0);
        const amount = 1 - distance / 4;
        for (let channel = 0; channel < CHANNELS; channel++) {
          const at = frame * CHANNELS + channel;
          frames[at] += amount * (baseline[at] - frames[at]);
        }
      }
      retained++;
    }
    first = last;
  }
  return retained;
}

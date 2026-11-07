import { FPS } from './index.mjs';

const middle = (phone) => (phone.start + phone.end) / 2;

export function warpSourceTime(warps, time) {
  for (const { start, anchor, contact, end } of warps ?? []) {
    if (time <= start || time >= end) continue;
    return time <= anchor
      ? start + (time - start) * (contact - start) / (anchor - start)
      : contact + (time - anchor) * (end - contact) / (end - anchor);
  }
  return time;
}

export function closureWarps(guide, clip, source, target) {
  if (!guide || target.length < 2) return undefined;
  const warps = [];
  for (const [i, phone] of target.entries()) {
    const points = guide.samples.get(phone)?.filter(({ confidence }) => confidence >= 0.85);
    if (!points?.length) continue;
    const peak = points.reduce((best, point) => point.confidence > best.confidence ? point : best);
    const left = Math.max(0, Math.min(target.length - 2, peak.time <= middle(phone) ? i - 1 : i));
    const phase = (peak.time - middle(target[left])) / (middle(target[left + 1]) - middle(target[left]));
    const anchor = middle(source[left]) + phase * (middle(source[left + 1]) - middle(source[left]));
    // Stops are often aligned to only 20 ms, but the lips can close before
    // that interval. Use nonoverlapping cells around source phone centers.
    const start = i > 0 ? (middle(source[i - 1]) + middle(source[i])) / 2 : source[i].start;
    const end = i + 1 < source.length ? (middle(source[i]) + middle(source[i + 1])) / 2 : source[i].end;
    if (anchor <= start || anchor >= end) continue;
    let contact, bestGap = 0.10;
    for (let frame = Math.ceil(start * FPS); frame / FPS <= end; frame++) {
      const time = frame / FPS;
      if (Math.abs(time - anchor) > 0.04) continue;
      // Bound both slopes, preserving source phone endpoints and time order.
      if ((time - start) / (anchor - start) < 0.5 || (time - start) / (anchor - start) > 2 ||
          (end - time) / (end - anchor) < 0.5 || (end - time) / (end - anchor) > 2) continue;
      const gap = guide.gapAt(clip, time);
      if (gap < bestGap) { bestGap = gap; contact = time; }
    }
    if (contact !== undefined) warps.push({ start, anchor, contact, end });
  }
  return warps.length ? warps : undefined;
}

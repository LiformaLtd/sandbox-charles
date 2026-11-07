// Threshold crossings are interpolated between 60 fps samples; interpolation
// does not increase the underlying temporal resolution.
export function contactEvents(gaps, fps = 60, threshold = 0.1) {
  if (!gaps.every(Number.isFinite)) throw new Error('Nonfinite lip gap track');
  const crossing = (frame) => (frame - 1 + (threshold - gaps[frame - 1]) / (gaps[frame] - gaps[frame - 1])) / fps;
  const events = [];
  let active;
  for (let frame = 0; frame < gaps.length; frame++) {
    if (gaps[frame] <= threshold) {
      if (!active) active = { start: frame === 0 ? 0 : crossing(frame), clippedStart: frame === 0,
        minimumGap: gaps[frame], deepest: frame / fps };
      if (gaps[frame] < active.minimumGap) { active.minimumGap = gaps[frame]; active.deepest = frame / fps; }
    } else if (active) {
      events.push({ ...active, end: crossing(frame), clippedEnd: false });
      active = undefined;
    }
  }
  if (active) events.push({ ...active, end: (gaps.length - 1) / fps, clippedEnd: true });
  return events;
}

export function associateContacts(phones, events, window = 0.2) {
  return associateFeatureEvents(phones, events, ['P', 'B', 'M'], window);
}

export function associateFeatureEvents(phones, events, labels, window = 0.2) {
  const eligible = new Set(labels);
  const labials = phones.map((phone, index) => ({ ...phone, index })).filter(({ phone }) => eligible.has(phone));
  const claims = new Map();
  const candidates = labials.map((phone) => {
    const ids = events.flatMap((event, i) => event.end >= phone.start - window && event.start <= phone.end + window ? [i] : []);
    for (const i of ids) claims.set(i, (claims.get(i) ?? 0) + 1);
    return ids;
  });
  return labials.map((phone, i) => {
    const ids = candidates[i];
    const status = !ids.length ? 'no_contact' : ids.length > 1 ? 'multiple_events'
      : claims.get(ids[0]) > 1 ? 'shared_or_neighbor_ambiguous'
        : events[ids[0]].clippedStart || events[ids[0]].clippedEnd ? 'clipped_event' : 'matched';
    const distance = (event) => Math.max(0, phone.start - event.end, event.start - phone.end);
    const ordered = [...ids].sort((a, b) => distance(events[a]) - distance(events[b]) ||
      Math.abs((events[a].start + events[a].end) - (phone.start + phone.end)) - Math.abs((events[b].start + events[b].end) - (phone.start + phone.end)));
    return { phone, status, candidates: ids.length, event: status === 'matched' ? events[ids[0]] : null,
      nearestCandidate: ordered.length ? events[ordered[0]] : null };
  });
}

export function offsetMeasurements(phone, event) {
  return {
    onsetFromStartMs: 1000 * (event.start - phone.start),
    releaseFromEndMs: 1000 * (event.end - phone.end),
    midpointFromPhoneMidpointMs: 500 * (event.start + event.end - phone.start - phone.end),
    deepestFromPhoneMidpointMs: 1000 * (event.deepest - (phone.start + phone.end) / 2),
    contactDurationMs: 1000 * (event.end - event.start),
  };
}

export function distribution(values) {
  if (!values.length) return { n: 0 };
  const ordered = [...values].sort((a, b) => a - b);
  const quantile = (fraction) => {
    const position = fraction * (ordered.length - 1), at = Math.floor(position);
    return ordered[at] + (ordered[Math.min(at + 1, ordered.length - 1)] - ordered[at]) * (position - at);
  };
  return { n: values.length, mean: values.reduce((a, b) => a + b, 0) / values.length, min: ordered[0],
    p05: quantile(.05), p10: quantile(.1), p25: quantile(.25), median: quantile(.5),
    p75: quantile(.75), p90: quantile(.9), p95: quantile(.95), max: ordered.at(-1) };
}

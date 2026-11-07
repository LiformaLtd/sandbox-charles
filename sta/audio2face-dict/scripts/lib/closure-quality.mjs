import { contactEvents, distribution } from './closure-events.mjs';

export function makeClosureProfile(report) {
  const phones = {};
  for (const phone of ['P', 'B', 'M']) {
    const stats = report.summary[phone].geometry.offsets;
    if (!(stats.onsetFromStartMs.n >= 30)) throw new Error('Insufficient native training contacts for ' + phone);
    phones[phone] = { examples: stats.onsetFromStartMs.n,
      onsetRangeMs: [stats.onsetFromStartMs.p05, stats.onsetFromStartMs.p95],
      releaseRangeMs: [stats.releaseFromEndMs.p05, stats.releaseFromEndMs.p95] };
  }
  return { version: 1, threshold: report.method.centerLipContactThreshold, fps: report.method.fps,
    trainingClips: report.trainingClips, trainingSetSha256: report.trainingSetSha256,
    population: report.method.population, percentileBand: [5, 95], phones };
}

export function validateClosureProfile(profile) {
  if (profile?.version !== 1 || profile.fps !== 60 || !Number.isFinite(profile.threshold) || profile.threshold <= 0) throw new Error('Invalid closure timing profile');
  for (const phone of ['P', 'B', 'M']) {
    const value = profile.phones?.[phone];
    if (!(value?.examples >= 30) || ![value.onsetRangeMs, value.releaseRangeMs].every((range) =>
      Array.isArray(range) && range.length === 2 && range.every(Number.isFinite) && range[0] <= range[1])) throw new Error('Invalid timing ranges for ' + phone);
  }
}

export function advanceWindow(phone, profile, duration) {
  const offsets = profile.phones[phone.phone];
  const start = phone.start + offsets.onsetRangeMs[0] / 1000;
  const end = phone.end + offsets.releaseRangeMs[1] / 1000;
  return { start: Math.max(0, start), end: Math.min(duration, end), clipped: start < 0 || end > duration || end <= start };
}

function associateWindows(phones, events, profile, duration) {
  const rows = phones.flatMap((phone, index) => profile.phones[phone.phone] ? [{ phone, index, window: advanceWindow(phone, profile, duration) }] : []);
  const claims = new Map();
  const candidates = rows.map(({ window }) => {
    const matches = events.flatMap((event, i) => window.end > window.start && event.end >= window.start && event.start <= window.end ? [i] : []);
    for (const index of matches) claims.set(index, (claims.get(index) ?? 0) + 1);
    return matches;
  });
  return rows.map((row, i) => {
    const ids = candidates[i];
    const status = row.window.clipped ? 'clipped_window' : !ids.length ? 'no_contact' : ids.length > 1 ? 'multiple_events'
      : claims.get(ids[0]) > 1 ? 'shared_or_neighbor_ambiguous'
        : events[ids[0]].clippedStart || events[ids[0]].clippedEnd ? 'clipped_event' : 'matched';
    return { ...row, status, event: status === 'matched' ? events[ids[0]] : null };
  });
}

export function checkAdvanceClosures(phones, actual, reference, profile) {
  validateClosureProfile(profile);
  if (actual.length !== reference.length) throw new Error('Closure tracks have different durations');
  const duration = (actual.length - 1) / profile.fps;
  const observed = associateWindows(phones, contactEvents(actual, profile.fps, profile.threshold), profile, duration);
  const expected = associateWindows(phones, contactEvents(reference, profile.fps, profile.threshold), profile, duration);
  return observed.map((value, i) => {
    const target = expected[i];
    const decidable = (status) => status === 'matched' || status === 'no_contact';
    const classification = !decidable(target.status) ? 'excluded_reference' : !decidable(value.status) ? 'excluded_prediction'
      : target.event ? value.event ? 'tp' : 'fn' : value.event ? 'fp' : 'tn';
    const timing = classification === 'tp' ? {
      onsetErrorMs: 1000 * (value.event.start - target.event.start),
      releaseErrorMs: 1000 * (value.event.end - target.event.end),
      durationErrorMs: 1000 * ((value.event.end - value.event.start) - (target.event.end - target.event.start)),
      depthError: value.event.minimumGap - target.event.minimumGap,
    } : null;
    const withinBand = (event) => {
      if (!event) return null;
      const bands = profile.phones[value.phone.phone];
      const onset = 1000 * (event.start - value.phone.start), release = 1000 * (event.end - value.phone.end);
      return { onset: onset >= bands.onsetRangeMs[0] && onset <= bands.onsetRangeMs[1],
        release: release >= bands.releaseRangeMs[0] && release <= bands.releaseRangeMs[1] };
    };
    return { phone: value.phone.phone, phoneIndex: value.index, start: value.phone.start, end: value.phone.end,
      window: value.window, classification, referenceStatus: target.status, predictionStatus: value.status,
      referenceContact: target.event, predictedContact: value.event,
      referenceWithinTrainingBand: withinBand(target.event), predictionWithinTrainingBand: withinBand(value.event), timing };
  });
}

export function summarizeAdvanceClosures(rows) {
  const counts = Object.fromEntries(['tp', 'fp', 'fn', 'tn', 'excluded_reference', 'excluded_prediction'].map((name) => [name, 0]));
  for (const row of rows) counts[row.classification]++;
  const ratio = (a, b) => b ? a / b : null;
  const timing = rows.filter((row) => row.timing).map((row) => row.timing);
  return { ...counts, totalPhonemes: rows.length,
    scoredPhonemes: counts.tp + counts.fp + counts.fn + counts.tn,
    precision: ratio(counts.tp, counts.tp + counts.fp), recall: ratio(counts.tp, counts.tp + counts.fn),
    f1: ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn),
    timingBandOutliers: { referenceOnset: rows.filter((row) => row.referenceWithinTrainingBand?.onset === false).length,
      referenceRelease: rows.filter((row) => row.referenceWithinTrainingBand?.release === false).length,
      predictionOnset: rows.filter((row) => row.predictionWithinTrainingBand?.onset === false).length,
      predictionRelease: rows.filter((row) => row.predictionWithinTrainingBand?.release === false).length },
    excludedReferenceReasons: Object.fromEntries([...new Set(rows.filter((row) => row.classification === 'excluded_reference').map((row) => row.referenceStatus))]
      .map((status) => [status, rows.filter((row) => row.classification === 'excluded_reference' && row.referenceStatus === status).length])),
    excludedPredictionReasons: Object.fromEntries([...new Set(rows.filter((row) => row.classification === 'excluded_prediction').map((row) => row.predictionStatus))]
      .map((status) => [status, rows.filter((row) => row.classification === 'excluded_prediction' && row.predictionStatus === status).length])),
    timing: Object.fromEntries(['onsetErrorMs','releaseErrorMs','durationErrorMs','depthError'].map((name) => [name, {
      signed: distribution(timing.map((value) => value[name])),
      absolute: distribution(timing.map((value) => Math.abs(value[name]))),
    }])),
  };
}

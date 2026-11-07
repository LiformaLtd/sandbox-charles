import { CHANNELS, FPS, SAMPLE_RATE, key, speechRuns, validatePhones } from './index.mjs';
import { chooseJoinedMatches } from './select-joins.mjs';

const midpoint = (phone) => (phone.start + phone.end) / 2;
const mix = (a, b, amount) => a + (b - a) * amount;
const fraction = (time, from, to) => (time - from) / (to - from);

function rankMatches(dictionary, phones, start, count, limit) {
  const candidates = dictionary.index[key(phones, start, count)] ?? [];
  const isolated = count === 1 && candidates.some((candidate) => candidate.isolated);
  const target = phones.slice(start, start + count);
  const ranked = [];
  for (const occurrence of candidates) {
    if (isolated && !occurrence.isolated) continue;
    const clip = dictionary.clips[occurrence.clip];
    const source = clip.phones.slice(occurrence.start, occurrence.start + count);
    let score = 0;
    if (count === 1) score = ((source[0].end - source[0].start) - (target[0].end - target[0].start)) ** 2;
    for (let i = 1; i < count; i++) {
      const sourceGap = midpoint(source[i]) - midpoint(source[i - 1]);
      const targetGap = midpoint(target[i]) - midpoint(target[i - 1]);
      score += (sourceGap - targetGap) ** 2;
    }
    const position = ranked.findIndex((candidate) => score < candidate.score);
    if (position >= 0) ranked.splice(position, 0, { clip, source, count, score });
    else if (ranked.length < limit) ranked.push({ clip, source, count, score });
    if (ranked.length > limit) ranked.pop();
  }
  return ranked;
}

function select(dictionary, phones, start, count) {
  return rankMatches(dictionary, phones, start, count, 1)[0];
}

function piece(match, index, from, to, first = index, last = index + 1) {
  return { match, from, to, first: match.source[first], last: match.source[last] };
}

function fillWithSingles(dictionary, phones, intervals, from) {
  for (const [i, interval] of intervals.entries()) {
    if (interval.kind !== 'miss') continue;
    // Borrow the pose at a neighboring match's anchor when possible so the
    // fallback starts and ends at the same poses as the surrounding motion.
    const previous = intervals[i - 1]?.pieces.at(-1);
    const next = intervals[i + 1]?.pieces[0];
    const outgoing = previous?.match ?? select(dictionary, phones, from + i, 1);
    const incoming = next?.match ?? select(dictionary, phones, from + i + 1, 1);
    if (!outgoing || !incoming) continue;
    const last = previous?.last ?? outgoing.source[0];
    const first = next?.first ?? incoming.source[0];
    interval.kind = 'fallback';
    interval.pieces = [
      { match: outgoing, from: midpoint(last), to: last.end, first: last, last },
      { match: incoming, from: first.start, to: midpoint(first), first, last: first },
    ];
  }
}

function runSegments(dictionary, phones, { from, to }, singleFallback, joinAware) {
  const first = phones[from], last = phones[to - 1];
  if (to - from === 1) {
    const match = select(dictionary, phones, from, 1);
    const source = match?.source[0];
    return [
      { start: first.start, end: midpoint(first), kind: match ? 'single' : 'miss',
        pieces: match ? [piece(match, 0, source.start, midpoint(source), 0, 0)] : [] },
      { start: midpoint(first), end: first.end, kind: match ? 'single' : 'miss',
        pieces: match ? [piece(match, 0, midpoint(source), source.end, 0, 0)] : [] },
    ];
  }
  const intervals = phones.slice(from, to - 1).map((phone, i) => ({
    start: midpoint(phone), end: midpoint(phones[from + i + 1]),
    first: phone, last: phones[from + i + 1], kind: 'miss', pieces: [],
  }));
  const coveredByTriple = new Uint8Array(intervals.length);
  const units = [];
  const limit = joinAware ? 8 : 1;
  for (let i = from; i + 1 < to; i++) {
    const matches = i + 2 < to ? rankMatches(dictionary, phones, i, 3, limit) : [];
    if (matches.length) {
      units.push({ start: i, count: 3, matches });
      coveredByTriple[i - from] = coveredByTriple[i - from + 1] = 1;
    } else if (!coveredByTriple[i - from]) {
      const pairs = rankMatches(dictionary, phones, i, 2, limit);
      if (pairs.length) units.push({ start: i, count: 2, matches: pairs });
    }
  }

  const selected = joinAware ? chooseJoinedMatches(units, dictionary.weights) : units.map(({ matches }) => matches[0]);
  for (const [i, match] of selected.entries()) {
    const start = units[i].start - from;
    for (let j = 0; j + 1 < match.count; j++) {
      const interval = intervals[start + j];
      interval.kind = match.count === 3 ? 'triple' : 'pair';
      interval.pieces.push(piece(match, j, midpoint(match.source[j]), midpoint(match.source[j + 1])));
    }
  }
  if (singleFallback) fillWithSingles(dictionary, phones, intervals, from);
  const opening = intervals[0].pieces[0];
  const closing = intervals.at(-1).pieces.at(-1);
  return [
    { start: first.start, end: midpoint(first), kind: 'edge',
      pieces: opening ? [piece(opening.match, 0, opening.first.start, opening.from, 0, 0)] : [] },
    ...intervals,
    { start: midpoint(last), end: last.end, kind: 'edge',
      pieces: closing ? [piece(closing.match, 0, closing.to, closing.last.end,
        closing.match.count - 1, closing.match.count - 1)] : [] },
  ];
}

export function planAnimation(dictionary, phones, pcm, { singleFallback = true, joinAware = true } = {}) {
  const duration = pcm.length / SAMPLE_RATE;
  validatePhones(phones, duration);
  const segments = [];
  const joins = [];
  const runs = speechRuns(phones, pcm, dictionary.silenceRms);
  const coverage = { triple: 0, pair: 0, single: 0, fallback: 0, miss: 0 };
  let end = 0;
  let pause;
  for (const run of runs) {
    const parts = runSegments(dictionary, phones, run, singleFallback, joinAware);
    if (parts[0].start > end) segments.push({ start: end, end: parts[0].start, pieces: [], pause });
    segments.push(...parts);
    if (run.to - run.from === 1) coverage[parts[0].kind]++;
    else for (const part of parts.slice(1, -1)) coverage[part.kind]++;
    end = parts.at(-1).end;
    pause = run.pause;
  }
  if (end < duration) segments.push({ start: end, end: duration, pieces: [] });

  for (let i = 1; i < segments.length; i++) {
    const left = segments[i - 1], right = segments[i];
    if (left.kind === 'fallback' || right.kind === 'fallback') continue;
    const outgoing = left.pieces.at(-1), incoming = right.pieces[0];
    if (!outgoing || !incoming || outgoing.match === incoming.match || !left.last || !right.first ||
        (outgoing.match.count !== 2 && incoming.match.count !== 2)) continue;
    const phone = left.last;
    const anchor = left.end;
    const outHalf = outgoing.last.end - outgoing.to;
    const inHalf = incoming.from - incoming.first.start;
    if (!(outHalf > 0 && inHalf > 0)) continue;
    const sourceEnd = (outgoing.match.clip.frameCount - 1) / FPS;
    const sourceAfter = (sourceEnd - outgoing.to) * (phone.end - anchor) / outHalf;
    const start = anchor - Math.min(0.025, (anchor - left.start) / 4, anchor - phone.start);
    const finish = anchor + Math.max(0, Math.min(0.025, (right.end - anchor) / 4, phone.end - anchor, sourceAfter));
    if (finish <= start) continue;
    joins.push({ start, end: finish, anchor, phone, left, right, outgoing, incoming,
      outClip: outgoing.match.clip.id, inClip: incoming.match.clip.id });
  }
  return { segments, joins, coverage, duration };
}

function addFrame(out, weights, clip, time, weight) {
  if (!weight) return;
  const frame = Math.max(0, Math.min(clip.frameCount - 1, time * FPS));
  const before = Math.floor(frame);
  const after = Math.min(clip.frameCount - 1, before + 1);
  const left = (clip.frameOffset + before) * CHANNELS;
  const right = (clip.frameOffset + after) * CHANNELS;
  for (let i = 0; i < CHANNELS; i++) out[i] += weight * mix(weights[left + i], weights[right + i], frame - before);
}

export function joinSourceTimes(join, time) {
  const { anchor, phone, outgoing, incoming, left, right } = join;
  if (time <= anchor) {
    return [
      mix(outgoing.from, outgoing.to, fraction(time, left.start, anchor)),
      mix(incoming.first.start, incoming.from, fraction(time, phone.start, anchor)),
    ];
  }
  return [
    mix(outgoing.to, outgoing.last.end, fraction(time, anchor, phone.end)),
    mix(incoming.from, incoming.to, fraction(time, anchor, right.end)),
  ];
}

function segmentIndex(segments, time) {
  let low = 0, high = segments.length - 1;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (time >= segments[mid].end) low = mid + 1;
    else high = mid;
  }
  return low;
}

export function bakeAnimation(dictionary, plan) {
  const frames = new Float32Array(Math.ceil(plan.duration * FPS) * CHANNELS);
  const { joins, segments } = plan;
  function sample(out, time, scale, join) {
    if (join && time > join.start && time < join.end) {
      const [left, right] = joinSourceTimes(join, time);
      const amount = fraction(time, join.start, join.end);
      addFrame(out, dictionary.weights, join.outgoing.match.clip, left, scale * (1 - amount));
      addFrame(out, dictionary.weights, join.incoming.match.clip, right, scale * amount);
      return;
    }
    const i = segmentIndex(segments, time);
    const { start, end, pieces, pause, kind, first, last } = segments[i];
    if (kind === 'fallback') {
      const [outgoing, incoming] = pieces;
      const amount = fraction(time, start, end);
      const outPhase = Math.min(1, fraction(time, start, first.end));
      const inPhase = Math.max(0, fraction(time, last.start, end));
      addFrame(out, dictionary.weights, outgoing.match.clip, mix(outgoing.from, outgoing.to, outPhase), scale * (1 - amount));
      addFrame(out, dictionary.weights, incoming.match.clip, mix(incoming.from, incoming.to, inPhase), scale * amount);
      return;
    }
    if (pieces.length) {
      const amount = fraction(time, start, end);
      for (const [j, item] of pieces.entries()) {
        const weight = pieces.length === 1 ? 1 : j === 0 ? 1 - amount : amount;
        addFrame(out, dictionary.weights, item.match.clip, mix(item.from, item.to, amount), weight * scale);
      }
      return;
    }
    const left = segments[i - 1]?.pieces.at(-1);
    const right = segments[i + 1]?.pieces[0];
    const quietFrom = pause?.start ?? start;
    const quietTo = pause?.end ?? end;
    const fade = Math.min(0.025, (quietTo - quietFrom) / 4);
    if (left && time < quietFrom + fade) {
      const amount = Math.max(0, fraction(time, quietFrom, quietFrom + fade));
      addFrame(out, dictionary.weights, left.match.clip, left.to, scale * (1 - amount));
    } else if (right && time > quietTo - fade) {
      const amount = Math.min(1, fraction(time, quietTo - fade, quietTo));
      addFrame(out, dictionary.weights, right.match.clip, right.from, scale * amount);
    }
  }

  let joinIndex = 0;
  const radius = 0.5 / FPS;
  for (let frame = 0; frame < frames.length / CHANNELS; frame++) {
    const time = frame / FPS;
    const out = frames.subarray(frame * CHANNELS, (frame + 1) * CHANNELS);
    while (joins[joinIndex]?.end <= time - radius) joinIndex++;
    const join = joins[joinIndex];
    const short = join && (Math.floor(join.start * FPS) + 1) / FPS >= join.end;
    if (short && time + radius > join.start && time - radius < join.end) {
      const start = Math.max(0, time - radius), end = Math.min(plan.duration, time + radius);
      const breaks = [start, ...[join.start, join.anchor, join.end].filter((point) => point > start && point < end), end];
      for (let i = 1; i < breaks.length; i++) {
        const span = breaks[i] - breaks[i - 1];
        // Two samples integrate the product of linear poses and linear fade weights.
        const center = (breaks[i - 1] + breaks[i]) / 2;
        const delta = span / Math.sqrt(12);
        sample(out, center - delta, span / (2 * (end - start)), join);
        sample(out, center + delta, span / (2 * (end - start)), join);
      }
    } else sample(out, time, 1, join);
  }
  return frames;
}

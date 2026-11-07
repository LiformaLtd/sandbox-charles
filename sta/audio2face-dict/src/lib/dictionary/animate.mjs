import { CHANNELS, FPS, SAMPLE_RATE, key, speechRuns, validatePhones } from './index.mjs';
import { chooseJoinedMatches } from './select-joins.mjs';
import { makeClosureGuide, closureCandidateCost, closureBlend, CONTACT_GAP } from './closures.mjs';
import { acousticDistance, describeSpeech } from './acoustics.mjs';
import { closureWarps, warpSourceTime } from './retime-closures.mjs';
import { predictLipTargets, predictedLipCost } from './selection-model.mjs';
import { retainAdvanceContacts } from './retain-contacts.mjs';
import { makeOpeningContext, openingSourceTime } from './opening-context.mjs';
import { boundedContextEnabled, makeBoundedContexts, applyBoundedContexts } from './bounded-context.mjs';
import { applyRecordedPauses } from './recorded-pauses.mjs';
import { applySmoothNeutral } from './neutral-transition.mjs';
import { DEFAULT_VISEME_PAIRS, substitutionMap } from './viseme-pairs.mjs';

const midpoint = (phone) => (phone.start + phone.end) / 2;
const mix = (a, b, amount) => a + (b - a) * amount;
const fraction = (time, from, to) => (time - from) / (to - from);

function rankMatches(dictionary, phones, start, count, limit, guide, optimization, alternatives) {
  const candidates = alternatives ?? dictionary.index[key(phones, start, count)] ?? [];
  const isolated = count === 1 && candidates.some((candidate) => candidate.isolated);
  const target = phones.slice(start, start + count);
  const ranked = [];
  for (const occurrence of candidates) {
    if (isolated && !occurrence.isolated) continue;
    const clip = dictionary.clips[occurrence.clip];
    const source = clip.phones.slice(occurrence.start, occurrence.start + count);
    let score = occurrence.approximate ? 0.040 ** 2 * (occurrence.approximate.position === 1 ? 0.5 : 0.25) : 0;
    if (count === 1) score = ((source[0].end - source[0].start) - (target[0].end - target[0].start)) ** 2;
    for (let i = 1; i < count; i++) {
      const sourceGap = midpoint(source[i]) - midpoint(source[i - 1]);
      const targetGap = midpoint(target[i]) - midpoint(target[i - 1]);
      score += (sourceGap - targetGap) ** 2;
    }
    if (optimization?.acoustics && clip.acoustics) {
      if (optimization.targets) {
        score += optimization.acousticWeight * 0.040 ** 2 * predictedLipCost(dictionary, clip, source, optimization.targets.slice(start, start + count));
      } else {
        for (let i = 0; i < count; i++) {
          score += optimization.acousticWeight * 0.040 ** 2 / count * acousticDistance(clip.acoustics[occurrence.start + i], optimization.acoustics[start + i]);
          score += 0.2 / count * ((source[i].end - source[i].start) - (target[i].end - target[i].start)) ** 2;
        }
      }
    }
    const warps = optimization?.closureRetiming ? closureWarps(guide, clip, source, target) : undefined;
    if (guide) score += 4 * 0.040 ** 2 * closureCandidateCost(guide, clip, source, target, warps);
    const position = ranked.findIndex((candidate) => score < candidate.score);
    const approximate = occurrence.approximate ? { ...occurrence.approximate, targetStart: target[0].start, targetEnd: target.at(-1).end } : undefined;
    const match = { clip, source, count, score, warps, approximate };
    if (position >= 0) ranked.splice(position, 0, match);
    else if (ranked.length < limit) ranked.push(match);
    if (ranked.length > limit) ranked.pop();
  }
  return ranked;
}

function similarTriples(dictionary, phones, start, substitutions) {
  const target = phones.slice(start, start + 3).map(({ phone }) => phone);
  const alternatives = [];
  for (let position = 0; position < 3; position++) {
    const from = target[position];
    const substitution = substitutions.get(from);
    if (!substitution) continue;
    const {to,pair} = substitution;
    const source = [...target];
    source[position] = to;
    for (const occurrence of dictionary.index[source.join(' ')] ?? []) {
      alternatives.push({ ...occurrence, approximate: { position, from, to, pair, target: target.join(' '), source: source.join(' ') } });
    }
  }
  return alternatives;
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

function runSegments(dictionary, phones, { from, to }, singleFallback, joinAware, guide, optimization) {
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
    const matches = i + 2 < to ? rankMatches(dictionary, phones, i, 3, limit, guide, optimization) : [];
    if (matches.length) {
      units.push({ start: i, count: 3, matches });
      coveredByTriple[i - from] = coveredByTriple[i - from + 1] = 1;
    } else if (!coveredByTriple[i - from]) {
      const pairs = rankMatches(dictionary, phones, i, 2, limit, guide, optimization);
      const approximate = optimization.visemeFallback && i + 2 < to
        ? rankMatches(dictionary, phones, i, 3, limit, guide, optimization, similarTriples(dictionary, phones, i, optimization.substitutions)) : [];
      const choices = [...approximate, ...pairs];
      // Compare the cost per contributed interval. The substitution penalty
      // lets an exact double win when its timing/joins are a better fit.
      if (approximate.length) {
        for (const match of choices) match.selectionScore = match.score / (match.count - 1);
        choices.sort((a, b) => a.selectionScore - b.selectionScore);
      }
      if (choices.length) units.push({ start: i, count: 2, matches: choices });
    }
  }

  const selected = joinAware ? chooseJoinedMatches(units, dictionary.weights, dictionary.lipRig,
    units.some(({ matches }) => matches.some((match) => match.approximate)) ? to - 1 : undefined) : units.map(({ matches }) => matches[0]);
  for (const [i, match] of selected.entries()) {
    const start = units[i].start - from;
    for (let j = 0; j + 1 < match.count; j++) {
      const interval = intervals[start + j];
      interval.pieces.push(piece(match, j, midpoint(match.source[j]), midpoint(match.source[j + 1])));
      interval.kind = interval.pieces.some(({ match }) => match.approximate) ? 'approximate'
        : interval.pieces.some(({ match }) => match.count === 3) ? 'triple' : 'pair';
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

function matchUsage(segments) {
  const matches = new Set();
  const singles = new Set();
  for (const segment of segments) {
    if (segment.kind === 'fallback') {
      for (const [i, item] of segment.pieces.entries()) {
        const anchor = i === 0 ? segment.start : segment.end;
        const source = i === 0 ? item.last : item.first;
        singles.add(`${anchor}:${item.match.clip.id}:${source.start}:${source.end}`);
      }
    } else if (segment.kind !== 'edge') {
      for (const { match } of segment.pieces) matches.add(match);
    }
  }
  const usage = { triple: 0, pair: 0, single: singles.size, approximate: 0 };
  for (const { count, approximate } of matches) {
    usage[approximate ? 'approximate' : count === 3 ? 'triple' : count === 2 ? 'pair' : 'single']++;
  }
  return usage;
}

export function planAnimation(dictionary, phones, pcm, { singleFallback = true, joinAware = true, closureAware = true, acousticAware = false, closureRetiming = false, acousticWeight = 0.25, acousticMode = 'predicted', visemeFallback = true,
  advanceClosures = true, advanceSelection = advanceClosures, advanceBlending = advanceClosures, openingContext = true,
  boundedContext = boundedContextEnabled, contextStrength = .25, contextBlend = 'anchored', recordedPauses = false, smoothNeutral = true, neutralFadeSeconds = .2, visemePairs = DEFAULT_VISEME_PAIRS } = {}) {
  const duration = pcm.length / SAMPLE_RATE;
  validatePhones(phones, duration);
  const guide = closureAware ? makeClosureGuide(dictionary, phones, pcm, advanceBlending) : undefined;
  const matchGuide = advanceSelection === advanceBlending ? guide : closureAware ? makeClosureGuide(dictionary, phones, pcm, advanceSelection) : undefined;
  const useAcoustics = acousticAware && (acousticMode === 'similarity' || dictionary.selectionModel) && dictionary.clips.some((clip) => clip.acoustics);
  const optimization = { acoustics: useAcoustics ? describeSpeech(pcm, phones) : undefined, closureRetiming, acousticWeight, visemeFallback, substitutions:substitutionMap(visemePairs) };
  if (optimization.acoustics && acousticMode === 'predicted' && dictionary.selectionModel) optimization.targets = predictLipTargets(dictionary.selectionModel, phones, optimization.acoustics);
  const segments = [];
  const joins = [];
  const runs = speechRuns(phones, pcm, dictionary.silenceRms);
  const coverage = { triple: 0, pair: 0, single: 0, fallback: 0, miss: 0, approximate: 0 };
  let end = 0;
  let pause;
  for (const run of runs) {
    const parts = runSegments(dictionary, phones, run, singleFallback, joinAware, matchGuide, optimization);
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
  const approximateMatches = [...new Set(segments.flatMap(({ pieces }) => pieces.map(({ match }) => match)).filter((match) => match.approximate))]
    .map(({ approximate, clip }) => ({ ...approximate, clip: clip.id }));
  const baselinePlan = guide?.advance ? planAnimation(dictionary, phones, pcm, {
    singleFallback, joinAware, closureAware, acousticAware, closureRetiming, acousticWeight, acousticMode, visemeFallback, visemePairs, openingContext,
    advanceClosures: false, advanceSelection: false, advanceBlending: false, boundedContext: false, smoothNeutral: false,
  }) : undefined;
  const leadingContext = openingContext ? makeOpeningContext(dictionary, phones, segments) : undefined;
  const plan = { segments, joins, coverage, usage: matchUsage(segments), approximateMatches, duration, guide, baselinePlan, leadingContext, phones, runs };
  plan.contexts = boundedContext ? makeBoundedContexts(dictionary, plan, { strength: contextStrength, mode: contextBlend }) : [];
  plan.contextBlend = contextBlend;
  plan.recordedPauses = recordedPauses;
  plan.frameCount = Math.ceil(pcm.length * FPS / SAMPLE_RATE);
  plan.smoothNeutral = smoothNeutral;
  plan.neutralFadeSeconds = neutralFadeSeconds;
  return plan;
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

function preserveClosures(dictionary, guide, frames) {
  const pose = new Float32Array(CHANNELS);
  for (const [i, phone] of guide.phones.entries()) {
    const points = guide.samples.get(phone);
    const confidenceThreshold = guide.advance ? .95 : .85;
    if (!points?.some(({ confidence }) => confidence >= confidenceThreshold)) continue;
    if (points.some(({ time }) => guide.frameGap(frames, Math.round(time * FPS) * CHANNELS) <= CONTACT_GAP)) continue;
    const previous = guide.phones[i - 1]?.phone;
    const next = guide.phones[i + 1]?.phone;
    let best;
    for (const { time, confidence } of points) {
      if (confidence < confidenceThreshold) continue;
      const frame = Math.round(time * FPS);
      const gap = guide.frameGap(frames, frame * CHANNELS);
      if (gap > 0.30) continue;
      for (const occurrence of dictionary.index[phone.phone] ?? []) {
        const clip = dictionary.clips[occurrence.clip];
        const source = clip.phones[occurrence.start];
        const preceding = previous && clip.phones[occurrence.start - 1]?.phone === previous;
        const following = next && clip.phones[occurrence.start + 1]?.phone === next;
        if (!preceding && !following) continue;
        const band = guide.advance ? guide.profile.phones[source.phone] : undefined;
        const start = Math.max(0, Math.ceil((source.start + (band ? band.onsetRangeMs[0] / 1000 : -0.02)) * FPS));
        const end = Math.min(clip.frameCount, Math.ceil((source.end + (band ? band.releaseRangeMs[1] / 1000 : 0.02)) * FPS));
        for (let sample = start; sample < end; sample++) {
          const sourceGap = guide.gapAt(clip, sample / FPS);
          if (sourceGap > 0.07) continue;
          const amount = (gap - 0.095) / (gap - sourceGap);
          if (amount > 0.8) continue;
          const timingError = Math.abs((sample / FPS - source.start) - (time - phone.start));
          const cost = amount + 6 * timingError - confidence - 0.15 * Number(preceding && following);
          if (!best || cost < best.cost) best = { frame, clip, sample, amount, cost };
        }
      }
    }
    if (!best) continue;
    // Restore contact with a recorded pose for the same sound and shared
    // context. Ease into the pose without extrapolating the rig weights.
    pose.fill(0);
    addFrame(pose, dictionary.weights, best.clip, best.sample / FPS, 1);
    for (const offset of [-1, 0, 1]) {
      const frame = best.frame + offset;
      if (frame < 0 || frame * CHANNELS >= frames.length) continue;
      const amount = best.amount * (offset === 0 ? 1 : 0.35);
      for (let channel = 0; channel < CHANNELS; channel++) {
        const index = frame * CHANNELS + channel;
        frames[index] = mix(frames[index], pose[channel], amount);
      }
    }
  }
}

export function bakeAnimation(dictionary, plan) {
  const frames = new Float32Array((plan.frameCount ?? Math.ceil(plan.duration * FPS)) * CHANNELS);
  const { joins, segments, guide } = plan;
  function blend(out, time, scale, left, leftTime, right, rightTime, amount) {
    leftTime = warpSourceTime(left.warps, leftTime);
    rightTime = warpSourceTime(right.warps, rightTime);
    const adjusted = closureBlend(guide, time, left.clip, leftTime, right.clip, rightTime, amount);
    addFrame(out, dictionary.weights, left.clip, leftTime, scale * (1 - adjusted));
    addFrame(out, dictionary.weights, right.clip, rightTime, scale * adjusted);
  }
  function sample(out, time, scale, join) {
    if (plan.leadingContext && time < plan.leadingContext.targetEnd) {
      const context = plan.leadingContext;
      addFrame(out, dictionary.weights, context.match.clip, openingSourceTime(context, time), scale);
      return;
    }
    if (join && time > join.start && time < join.end) {
      const [left, right] = joinSourceTimes(join, time);
      const amount = fraction(time, join.start, join.end);
      blend(out, time, scale, join.outgoing.match, left, join.incoming.match, right, amount);
      return;
    }
    const i = segmentIndex(segments, time);
    const { start, end, pieces, pause, kind, first, last } = segments[i];
    if (kind === 'fallback') {
      const [outgoing, incoming] = pieces;
      const amount = fraction(time, start, end);
      const outPhase = Math.min(1, fraction(time, start, first.end));
      const inPhase = Math.max(0, fraction(time, last.start, end));
      blend(out, time, scale, outgoing.match, mix(outgoing.from, outgoing.to, outPhase),
        incoming.match, mix(incoming.from, incoming.to, inPhase), amount);
      return;
    }
    if (pieces.length) {
      const amount = fraction(time, start, end);
      const [left, right] = pieces;
      const leftTime = mix(left.from, left.to, amount);
      if (right) {
        blend(out, time, scale, left.match, leftTime, right.match, mix(right.from, right.to, amount), amount);
      } else addFrame(out, dictionary.weights, left.match.clip, warpSourceTime(left.match.warps, leftTime), scale);
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
  if (guide) preserveClosures(dictionary, guide, frames);
  if (plan.baselinePlan && guide?.advance) retainAdvanceContacts(frames, bakeAnimation(dictionary, plan.baselinePlan), guide);
  plan.contextDiagnostics = applyBoundedContexts(dictionary, frames, plan.contexts ?? [], { mode: plan.contextBlend });
  if (plan.recordedPauses) plan.pauseDiagnostics = applyRecordedPauses(dictionary,frames,plan);
  if (plan.smoothNeutral) plan.neutralDiagnostics = applySmoothNeutral(dictionary,frames,plan,{seconds:plan.neutralFadeSeconds});
  return frames;
}

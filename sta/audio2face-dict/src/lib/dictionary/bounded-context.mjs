import profile from '../../../data/phoneme-context-candidates.json' with { type: 'json' };
import channels from '../studio/channels.json' with { type: 'json' };
import { CHANNELS, FPS } from './index.mjs';

export const boundedContextEnabled = profile.active;
const midpoint = (phone) => (phone.start + phone.end) / 2;
const featureChannels = {
  rounding: ['mouthFunnel','mouthPucker','mouthStretchLeft','mouthStretchRight','mouthRollLower','mouthRollUpper'],
  jawOpening: ['jawOpen','jawForward','mouthLowerDownLeft','mouthLowerDownRight','mouthUpperUpLeft','mouthUpperUpRight'],
  fvProximity: ['mouthRollLower','mouthShrugLower','mouthUpperUpLeft','mouthUpperUpRight','mouthPressLeft','mouthPressRight','mouthClose'],
  lipClosure: ['mouthClose','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthPressLeft','mouthPressRight'],
};
const smoothstep = (value) => { const x = Math.max(0, Math.min(1, value)); return x * x * (3 - 2 * x); };

export function contextSourceTime(context, time) {
  return context.sourceAnchor + (time - context.targetAnchor) * context.slope;
}

export function contextWeight(context, time) {
  if (time <= context.start || time >= context.end) return 0;
  const phase = (time - context.start) / (context.end - context.start);
  return context.strength * Math.sin(Math.PI * phase) ** 2;
}

export function makeBoundedContexts(dictionary, plan, { candidates = profile, strength = .25, mode = 'anchored' } = {}) {
  if (!['absolute','anchored'].includes(mode)) throw new RangeError('Unknown context blend mode');
  if (!Number.isFinite(strength) || strength < 0 || strength > .5) throw new RangeError('Context strength must be between 0 and 0.5');
  if (!dictionary.lipRig || !plan.phones || !plan.runs) return [];
  const selected = new Map();
  for (const segment of plan.segments) {
    if (!segment.first || segment.kind === 'fallback') continue;
    for (const piece of segment.pieces) {
      const start = plan.phones.indexOf(segment.first) - piece.match.source.indexOf(piece.first);
      if (start >= 0 && piece.match.count >= 2) selected.set(piece.match, start);
    }
  }
  const contexts = [];
  for (const [match, at] of selected) {
    const last = at + match.count - 1;
    const sourceStart = match.clip.phones.indexOf(match.source[0]);
    const sourceLast = sourceStart + match.count - 1;
    const runIndex = plan.runs.findIndex((run) => at >= run.from && last < run.to);
    if (runIndex < 0 || sourceStart < 0) continue;
    const run = plan.runs[runIndex];
    const sourceRun = (match.clip.runs ?? [{ from: 0, to: match.clip.phones.length }]).find((r) => sourceStart >= r.from && sourceLast < r.to);
    if (!sourceRun) continue;
    for (const side of ['before','after']) {
      const first = side === 'before', targetPhone = plan.phones[first ? at : last], sourcePhone = match.source[first ? 0 : match.count - 1];
      const targetProfile = candidates.phonemes[targetPhone.phone], sourceProfile = candidates.phonemes[sourcePhone.phone];
      const property = first ? 'candidateBeforeMs' : 'candidateAfterMs';
      if (!targetProfile?.evidence?.length || !sourceProfile?.evidence?.length || !(targetProfile[property] > 0 && sourceProfile[property] > 0)) continue;
      const neighbor = first ? at - 1 : last + 1, sourceNeighbor = first ? sourceStart - 1 : sourceLast + 1;
      const targetHasNeighbor = neighbor >= run.from && neighbor < run.to;
      const sourceHasNeighbor = sourceNeighbor >= sourceRun.from && sourceNeighbor < sourceRun.to;
      // Keep the dedicated utterance-opening fix. Internal context needs a
      // real adjacent source phoneme, not padded silence from another clip.
      if (first && (!targetHasNeighbor || !sourceHasNeighbor)) continue;
      if (!first && targetHasNeighbor !== sourceHasNeighbor) continue;
      // Outside a selected n-gram, its donor neighbor is not guaranteed to
      // match the target. A time bound alone cannot prevent the wrong gesture.
      if (mode !== 'absolute' && targetHasNeighbor && sourceHasNeighbor) {
        const a = plan.phones[neighbor].phone, b = match.clip.phones[sourceNeighbor].phone;
        if (a !== b && !(['P','B'].includes(a) && ['P','B'].includes(b))) continue;
      }
      const targetAnchor = midpoint(targetPhone), sourceAnchor = midpoint(sourcePhone);
      const targetGap = mode !== 'absolute' && targetHasNeighbor ? Math.abs(midpoint(plan.phones[neighbor]) - targetAnchor) : first ? midpoint(plan.phones[at + 1]) - targetAnchor : targetAnchor - midpoint(plan.phones[last - 1]);
      const sourceGap = mode !== 'absolute' && sourceHasNeighbor ? Math.abs(midpoint(match.clip.phones[sourceNeighbor]) - sourceAnchor) : first ? midpoint(match.source[1]) - sourceAnchor : sourceAnchor - midpoint(match.source.at(-2));
      const slope = Math.max(.5, Math.min(2, sourceGap / targetGap));
      let start, end;
      if (first) {
        const sourceBound = Math.max(sourcePhone.start - sourceProfile[property] / 1000,
          midpoint(match.clip.phones[sourceNeighbor]), 0);
        start = Math.max(targetPhone.start - targetProfile[property] / 1000,
          midpoint(plan.phones[neighbor]), targetAnchor + (sourceBound - sourceAnchor) / slope,
          plan.runs[runIndex - 1]?.pause?.end ?? 0);
        end = targetAnchor;
      } else {
        const sourceBound = Math.min(sourcePhone.end + sourceProfile[property] / 1000,
          sourceHasNeighbor ? midpoint(match.clip.phones[sourceNeighbor]) : Infinity,
          (match.clip.frameCount - 1) / FPS,
          sourceRun.to < match.clip.phones.length ? match.clip.phones[sourceRun.to - 1].end : Infinity);
        start = targetAnchor;
        end = Math.min(targetPhone.end + targetProfile[property] / 1000,
          targetHasNeighbor ? midpoint(plan.phones[neighbor]) : plan.duration,
          targetAnchor + (sourceBound - sourceAnchor) / slope, run.pause?.start ?? plan.duration);
      }
      if (!(end - start >= 1 / FPS)) continue;
      const attribution = mode === 'absolute' ? 1 : Math.min(...targetProfile.evidence.map(e => e.matchedFraction ?? 1), ...sourceProfile.evidence.map(e => e.matchedFraction ?? 1));
      const reliability = attribution * (targetProfile.thresholdSensitive ? .5 : 1) * (targetProfile.broadContext ? .75 : 1);
      const poseChannels = [...new Set(targetProfile.evidence.flatMap(({ feature }) => featureChannels[feature] ?? []))].map((name) => channels.indexOf(name));
      if (!poseChannels.length) continue;
      contexts.push({ match, side, phone: targetPhone.phone, start, end, targetAnchor, sourceAnchor, slope, strength: strength * reliability, poseChannels,
        attribution, targetNeighbor: targetHasNeighbor ? plan.phones[neighbor].phone : null,
        sourceNeighbor: sourceHasNeighbor ? match.clip.phones[sourceNeighbor].phone : null });
    }
  }
  return contexts;
}

export function applyBoundedContexts(dictionary, frames, contexts, { diagnostics = {}, mode = 'anchored' } = {}) {
  if (!['absolute', 'anchored'].includes(mode)) throw new RangeError('Unknown context blend mode');
  if (!contexts.length) return diagnostics;
  const center = Math.floor(dictionary.lipRig.neutral.length / 2);
  const basis = dictionary.lipRig.deltas.map((row) => row[center]);
  const gapAt = (pose) => basis.reduce((gap, value, i) => gap + value * pose[i], dictionary.lipRig.neutral[center]);
  const base = new Float32Array(CHANNELS), donor = new Float32Array(CHANNELS), channelWeight = new Float32Array(CHANNELS), effective = new Float32Array(CHANNELS);
  // Align each donor's pose to the unmodified assembled pose at its phoneme
  // anchor. Transfer the surrounding motion without importing its pose bias.
  // Compute every offset before writing frames so overlapping contexts cannot
  // feed back into one another or depend on iteration order.
  const offsets = contexts.map(context => {
    const offset = new Float32Array(CHANNELS);
    if (mode === 'absolute') return offset;
    const sourceFrame = context.sourceAnchor * FPS;
    const targetFrame = Math.max(0, Math.min(frames.length / CHANNELS - 1, context.targetAnchor * FPS));
    if (sourceFrame < 0 || sourceFrame > context.match.clip.frameCount - 1) throw new Error('Context sampled outside donor clip');
    const sf = Math.floor(sourceFrame), tf = Math.floor(targetFrame);
    const sa = (context.match.clip.frameOffset + sf) * CHANNELS;
    const sb = (context.match.clip.frameOffset + Math.min(sf + 1, context.match.clip.frameCount - 1)) * CHANNELS;
    const ta = tf * CHANNELS, tb = Math.min(tf + 1, frames.length / CHANNELS - 1) * CHANNELS;
    for (const c of context.poseChannels) {
      const source = dictionary.weights[sa + c] + (dictionary.weights[sb + c] - dictionary.weights[sa + c]) * (sourceFrame - sf);
      const target = frames[ta + c] + (frames[tb + c] - frames[ta + c]) * (targetFrame - tf);
      offset[c] = target - source;
    }
    return offset;
  });
  diagnostics.contexts = contexts.length; diagnostics.framesBlended = 0; diagnostics.contactLimitedFrames = 0;
  for (let frame = 0; frame < frames.length / CHANNELS; frame++) {
    const time = frame / FPS, out = frames.subarray(frame * CHANNELS, (frame + 1) * CHANNELS);
    donor.fill(0); channelWeight.fill(0); let total = 0;
    for (const [index, context] of contexts.entries()) {
      const weight = contextWeight(context, time);
      if (!weight) continue;
      const sourceFrame = contextSourceTime(context, time) * FPS;
      if (sourceFrame < 0 || sourceFrame > context.match.clip.frameCount - 1) throw new Error('Context sampled outside donor clip');
      const before = Math.floor(sourceFrame), after = Math.min(context.match.clip.frameCount - 1, before + 1), phase = sourceFrame - before;
      const a = (context.match.clip.frameOffset + before) * CHANNELS, b = (context.match.clip.frameOffset + after) * CHANNELS;
      for (const channel of context.poseChannels) {
        const pose = dictionary.weights[a + channel] + (dictionary.weights[b + channel] - dictionary.weights[a + channel]) * phase + offsets[index][channel];
        donor[channel] += weight * Math.max(0, Math.min(1, pose));
        channelWeight[channel] += weight;
      }
      total += weight;
    }
    if (!total) continue;
    base.set(out); effective.set(base);
    for (let channel = 0; channel < CHANNELS; channel++) {
      if (channelWeight[channel]) effective[channel] = base[channel] + Math.min(.5, channelWeight[channel]) * (donor[channel] / channelWeight[channel] - base[channel]);
    }
    const originalGap = gapAt(base), donorGap = gapAt(effective);
    // Preserve the current contact mask. Continuous taper prevents a closed
    // donor from introducing a new closure into an open target (or vice versa).
    const safety = originalGap <= .1 ? smoothstep((.1 - donorGap) / .05) : smoothstep((donorGap - .1) / .05);
    if (safety < 1) diagnostics.contactLimitedFrames++;
    if (!safety) continue;
    for (let channel = 0; channel < CHANNELS; channel++) out[channel] = base[channel] + safety * (effective[channel] - base[channel]);
    if ((gapAt(out) <= .1) !== (originalGap <= .1)) out.set(base); // Float32 boundary safeguard.
    else diagnostics.framesBlended++;
  }
  return diagnostics;
}

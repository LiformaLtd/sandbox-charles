import channels from '../studio/channels.json' with { type: 'json' };
import { CHANNELS, FPS } from './index.mjs';
import { warpSourceTime } from './retime-closures.mjs';

const mouthChannels = [
  ['jawOpen', 3], ['mouthClose', 3],
  ['mouthFunnel', 1], ['mouthPucker', 1],
  ['mouthRollLower', 1], ['mouthRollUpper', 1],
  ['mouthStretchLeft', 1], ['mouthStretchRight', 1],
  ['mouthLowerDownLeft', 1], ['mouthLowerDownRight', 1],
  ['mouthUpperUpLeft', 1], ['mouthUpperUpRight', 1],
].map(([name, weight]) => ({ index: channels.indexOf(name), weight }));

const channelWeight = mouthChannels.reduce((sum, { weight }) => sum + weight, 0);
const gapLandmarks = [2, 4, 6];
const midpoint = ({ start, end }) => (start + end) / 2;

// A shared phone uses its onset, midpoint, and release. A shared interval
// uses the same midpoint-to-midpoint mapping as the triple crossfade.
function signature(weights, match, start, overlap, gapBasis) {
  const first = match.source[start];
  const from = overlap === 1 ? first.start : midpoint(first);
  const middle = midpoint(first);
  const to = overlap === 1 ? first.end : midpoint(match.source[start + 1]);
  return [from, overlap === 1 ? middle : (from + to) / 2, to].map((time) => {
    const position = Math.max(0, Math.min(match.clip.frameCount - 1, warpSourceTime(match.warps, time) * FPS));
    const frame = Math.floor(position);
    const left = (match.clip.frameOffset + frame) * CHANNELS;
    const right = (match.clip.frameOffset + Math.min(frame + 1, match.clip.frameCount - 1)) * CHANNELS;
    const amount = position - frame;
    const values = mouthChannels.map(({ index }) =>
      weights[left + index] + (weights[right + index] - weights[left + index]) * amount);
    for (const deltas of gapBasis) {
      let gap = 0;
      for (const { index, delta } of deltas) {
        gap += delta * (weights[left + index] + (weights[right + index] - weights[left + index]) * amount);
      }
      values.push(gap);
    }
    return values;
  });
}

function transitionCost(left, right, featureWeights) {
  let pose = 0;
  let movement = 0;
  for (const [i, weight] of featureWeights.entries()) {
    for (let sample = 0; sample < left.length; sample++) {
      const difference = left[sample][i] - right[sample][i];
      pose += weight * difference ** 2;
      if (sample > 0) {
        const previousDifference = left[sample - 1][i] - right[sample - 1][i];
        movement += weight * (difference - previousDifference) ** 2;
      }
    }
  }
  // About a 0.10 blendshape difference or a 40 ms timing mismatch has
  // unit cost. Prefer matching motion as well as matching static poses.
  return pose / 3 + 0.5 * movement / 2;
}

export function chooseJoinedMatches(units, weights, lipRig, intervalEnd) {
  const gapBasis = lipRig ? gapLandmarks.map((landmark) =>
    lipRig.deltas.flatMap((values, index) =>
      values[landmark] ? [{ index, delta: values[landmark] }] : [])) : [];
  const featureWeights = mouthChannels.map(({ weight }) => weight / (channelWeight * 0.10 ** 2));
  // Measure lip separation directly in addition to the rig channel proxies.
  if (lipRig) featureWeights.push(...[1, 2, 1].map((weight) => weight / (4 * 0.60 ** 2)));
  let costs = [];
  const back = [];
  const signatures = new Map();
  const cachedSignature = (match, start, overlap) => {
    if (!signatures.has(match)) signatures.set(match, new Map());
    const cache = signatures.get(match), key = start + ':' + overlap;
    if (!cache.has(key)) cache.set(key, signature(weights, match, start, overlap, gapBasis));
    return cache.get(key);
  };
  for (const [i, unit] of units.entries()) {
    const previous = units[i - 1];
    const bestPrevious = costs.indexOf(Math.min(...costs));
    const predecessors = [];
    const nextCosts = unit.matches.map((match) => {
      const timing = (match.selectionScore ?? match.score) / 0.040 ** 2;
      if (!previous) {
        predecessors.push(bestPrevious);
        return timing + (costs[bestPrevious] ?? 0);
      }
      let winner = 0;
      let best = Infinity;
      for (const [j, prior] of previous.matches.entries()) {
        const overlap = Math.min(match.count, previous.start + prior.count - unit.start);
        const transition = overlap > 0 ? transitionCost(
          cachedSignature(prior, unit.start - previous.start, overlap),
          cachedSignature(match, 0, overlap), featureWeights) : 0;
        const uncovered = intervalEnd === undefined ? 0 : Math.max(0, unit.start - (previous.start + prior.count - 1));
        const score = costs[j] + transition + 4 * uncovered;
        if (score < best) {
          best = score;
          winner = j;
        }
      }
      predecessors.push(winner);
      return timing + best;
    });
    back.push(predecessors);
    costs = nextCosts;
  }

  const last = units.at(-1);
  const finalCosts = costs.map((cost, i) => cost + (intervalEnd === undefined ? 0 :
    4 * Math.max(0, intervalEnd - (last.start + last.matches[i].count - 1))));
  let choice = finalCosts.indexOf(Math.min(...finalCosts));
  const selected = new Array(units.length);
  for (let i = units.length - 1; i >= 0; i--) {
    selected[i] = units[i].matches[choice];
    choice = back[i][choice];
  }
  return selected;
}

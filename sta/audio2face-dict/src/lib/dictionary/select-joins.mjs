import channels from '../studio/channels.json' with { type: 'json' };
import { CHANNELS, FPS } from './index.mjs';

const mouthChannels = [
  ['jawOpen', 3], ['mouthClose', 3],
  ['mouthFunnel', 1], ['mouthPucker', 1],
  ['mouthRollLower', 1], ['mouthRollUpper', 1],
  ['mouthStretchLeft', 1], ['mouthStretchRight', 1],
  ['mouthLowerDownLeft', 1], ['mouthLowerDownRight', 1],
  ['mouthUpperUpLeft', 1], ['mouthUpperUpRight', 1],
].map(([name, weight]) => ({ index: channels.indexOf(name), weight }));

const channelWeight = mouthChannels.reduce((sum, { weight }) => sum + weight, 0);
const midpoint = ({ start, end }) => (start + end) / 2;

// A shared phone uses its onset, midpoint, and release. A shared interval
// uses the same midpoint-to-midpoint mapping as the triple crossfade.
function signature(weights, match, start, overlap) {
  const first = match.source[start];
  const from = overlap === 1 ? first.start : midpoint(first);
  const middle = midpoint(first);
  const to = overlap === 1 ? first.end : midpoint(match.source[start + 1]);
  return [from, overlap === 1 ? middle : (from + to) / 2, to].map((time) => {
    const position = Math.max(0, Math.min(match.clip.frameCount - 1, time * FPS));
    const frame = Math.floor(position);
    const left = (match.clip.frameOffset + frame) * CHANNELS;
    const right = (match.clip.frameOffset + Math.min(frame + 1, match.clip.frameCount - 1)) * CHANNELS;
    return mouthChannels.map(({ index }) =>
      weights[left + index] + (weights[right + index] - weights[left + index]) * (position - frame));
  });
}

function transitionCost(left, right) {
  let pose = 0;
  let movement = 0;
  for (const [i, { weight }] of mouthChannels.entries()) {
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
  return (pose / 3 + 0.5 * movement / 2) / (channelWeight * 0.10 ** 2);
}

export function chooseJoinedMatches(units, weights) {
  let costs = [];
  const back = [];
  for (const [i, unit] of units.entries()) {
    const previous = units[i - 1];
    const overlap = previous ? previous.start + previous.count - unit.start : 0;
    const linked = overlap > 0;
    const previousPoses = linked ? previous.matches.map((match) =>
      signature(weights, match, unit.start - previous.start, overlap)) : [];
    const bestPrevious = costs.indexOf(Math.min(...costs));
    const predecessors = [];
    const nextCosts = unit.matches.map((match) => {
      const timing = match.score / 0.040 ** 2;
      if (!linked) {
        predecessors.push(bestPrevious);
        return timing + (costs[bestPrevious] ?? 0);
      }
      const pose = signature(weights, match, 0, overlap);
      let winner = 0;
      let best = Infinity;
      for (const [j, prior] of previousPoses.entries()) {
        const score = costs[j] + transitionCost(prior, pose);
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

  let choice = costs.indexOf(Math.min(...costs));
  const selected = new Array(units.length);
  for (let i = units.length - 1; i >= 0; i--) {
    selected[i] = units[i].matches[choice];
    choice = back[i][choice];
  }
  return selected;
}

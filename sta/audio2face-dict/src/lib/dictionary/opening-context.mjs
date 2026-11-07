import { CHANNELS, FPS } from './index.mjs';

const vowels = new Set('AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW'.split(' '));

export function openingSourceTime(context, time) {
  const phase = Math.max(0, Math.min(1, time / context.targetEnd));
  return context.sourceFrom + phase * (context.sourceTo - context.sourceFrom);
}

export function makeOpeningContext(dictionary, phones, segments) {
  const first = phones[0], next = phones[1];
  if (!dictionary.lipRig || !vowels.has(first?.phone) || !['P','B','M'].includes(next?.phone) || first.start < 1 / FPS) return;
  const opening = segments.find((segment) => segment.kind === 'edge' && segment.start === first.start)?.pieces[0];
  if (!opening) return;
  const { match } = opening, sourceTo = opening.from;
  const { neutral, deltas } = dictionary.lipRig, center = Math.floor(neutral.length / 2);
  const gapAt = (time) => {
    const frame = Math.max(0, Math.min(match.clip.frameCount - 1, time * FPS));
    const left = (match.clip.frameOffset + Math.floor(frame)) * CHANNELS;
    const right = (match.clip.frameOffset + Math.min(match.clip.frameCount - 1, Math.floor(frame) + 1)) * CHANNELS;
    let gap = neutral[center];
    for (let channel = 0; channel < CHANNELS; channel++) gap += deltas[channel][center] *
      (dictionary.weights[left + channel] + (dictionary.weights[right + channel] - dictionary.weights[left + channel]) * (frame % 1));
    return gap;
  };
  if (gapAt(sourceTo) > .1) return; // The normal opening pose is already open.
  // Recover only the final continuous open run before this donor closure.
  // Earlier donor closures must not leak in from an unrelated preceding word.
  const earliest = Math.max(0, sourceTo - Math.min(.2, first.start * 2));
  let frame = Math.floor(sourceTo * FPS);
  while (frame / FPS >= earliest && gapAt(frame / FPS) <= .1) frame--;
  if (frame / FPS < earliest) return;
  while ((frame - 1) / FPS >= earliest && gapAt((frame - 1) / FPS) > .1) frame--;
  const sourceFrom = Math.max(earliest, frame / FPS);
  if (gapAt(sourceFrom) <= .1 || sourceFrom >= sourceTo) return;
  return { match, sourceFrom, sourceTo, targetEnd: first.start };
}

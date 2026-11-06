import type { TimedToken } from './types';

export function ctcForcedAlign(
  logProbs: Float32Array,
  timeSteps: number,
  vocabSize: number,
  phoneIds: number[],
  blankIdx = 0,
): number[] {
  const forcedPath: number[] = [blankIdx];
  for (const p of phoneIds) {
    forcedPath.push(p);
    forcedPath.push(blankIdx);
  }
  const S = forcedPath.length;
  const NEG_INF = -1e12;

  let alpha = new Float64Array(S);
  let newAlpha = new Float64Array(S);
  alpha.fill(NEG_INF);
  alpha[0] = getLP(logProbs, 0, blankIdx, vocabSize);
  if (S > 1) alpha[1] = getLP(logProbs, 0, forcedPath[1], vocabSize);

  const steps = new Uint8Array(timeSteps * S);

  for (let t = 1; t < timeSteps; t++) {
    for (let s = 0; s < S; s++) {
      let bestPrev = s;
      let bestVal = alpha[s];

      if (s > 0 && alpha[s - 1] > bestVal) {
        bestVal = alpha[s - 1];
        bestPrev = s - 1;
      }

      if (
        s > 1 &&
        forcedPath[s] !== blankIdx &&
        forcedPath[s] !== forcedPath[s - 2] &&
        alpha[s - 2] > bestVal
      ) {
        bestVal = alpha[s - 2];
        bestPrev = s - 2;
      }

      newAlpha[s] = bestVal + getLP(logProbs, t, forcedPath[s], vocabSize);
      steps[t * S + s] = s - bestPrev;
    }
    const previous = alpha;
    alpha = newAlpha;
    newAlpha = previous;
  }

  let endState = S - 1;
  if (S > 1 && alpha[S - 2] > alpha[S - 1]) endState = S - 2;

  const framePhoneIdx: number[] = new Array(timeSteps);
  let s = endState;
  for (let t = timeSteps - 1; t >= 0; t--) {
    framePhoneIdx[t] = s % 2 === 0 ? -1 : (s - 1) >> 1;
    s -= steps[t * S + s];
  }

  return framePhoneIdx;
}

function getLP(logProbs: Float32Array, t: number, v: number, vocab: number): number {
  return logProbs[t * vocab + v] ?? -1e12;
}

export function extractSegments(
  framePhoneIdx: number[],
  phones: string[],
  frameShiftS: number,
): TimedToken[] {
  if (framePhoneIdx.length === 0) return [];

  const out: TimedToken[] = [];
  let prev = framePhoneIdx[0];
  let startFrame = 0;

  for (let t = 1; t < framePhoneIdx.length; t++) {
    const idx = framePhoneIdx[t];
    if (idx !== prev) {
      out.push({
        word: prev >= 0 ? phones[prev] : '',
        start: startFrame * frameShiftS,
        end: t * frameShiftS,
      });
      startFrame = t;
      prev = idx;
    }
  }

  out.push({
    word: prev >= 0 ? phones[prev] : '',
    start: startFrame * frameShiftS,
    end: framePhoneIdx.length * frameShiftS,
  });

  return out;
}

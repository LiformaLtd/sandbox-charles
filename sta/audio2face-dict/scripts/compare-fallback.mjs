import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { bakeAnimation, planAnimation } from '../src/lib/dictionary/animate.mjs';
import { CHANNELS, FPS } from '../src/lib/dictionary/index.mjs';

const local = new URL('../.local/fresh/', import.meta.url);
const packed = new URL('../static/dictionary/', import.meta.url);
const json = async (url) => JSON.parse(await readFile(url));
const floats = async (url) => {
  const bytes = await readFile(url);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};

const dictionary = await json(new URL('index.json', packed));
assert.equal(dictionary.corpus, 'targeted', 'Pack the targeted corpus first');
dictionary.weights = await floats(new URL(dictionary.weightsFile, packed));
const [rows, rig] = await Promise.all([
  json(new URL('clips.json', local)),
  json(new URL('lip-rig.json', local)),
]);
const landmarks = rig.neutral.length;
const center = Math.floor(landmarks / 2);
const joinComparison = process.argv.includes('--join-aware');
const variants = joinComparison ? ['timingOnly', 'joinAware'] : ['without', 'with'];
const references = ['geometry', 'fitted'];
const threshold = 0.1;
const padding = 0.02;
const joinRadius = 0.035;

function gaps(track) {
  const out = new Float32Array(track.length / CHANNELS * landmarks);
  for (let frame = 0; frame < track.length / CHANNELS; frame++) {
    for (let j = 0; j < landmarks; j++) {
      let gap = rig.neutral[j];
      for (let i = 0; i < CHANNELS; i++) gap += track[frame * CHANNELS + i] * rig.deltas[i][j];
      out[frame * landmarks + j] = gap;
    }
  }
  return out;
}

function contact(track, from, to) {
  let first = null;
  let last = null;
  for (let frame = from; frame < to; frame++) {
    if (track[frame * landmarks + center] > threshold) continue;
    first ??= frame;
    last = frame;
  }
  return { first, last };
}

function makeMetrics() {
  return {
    squares: 0, values: 0,
    affectedSquares: 0, affectedValues: 0,
    joinSquares: 0, joinValues: 0,
    peakStep: 0, joinCount: 0, joinsOverReference: 0,
    closure: { tp: 0, fp: 0, fn: 0, tn: 0, onsetFrames: 0, releaseFrames: 0 },
    affectedClosure: { tp: 0, fp: 0, fn: 0, tn: 0, onsetFrames: 0, releaseFrames: 0 },
  };
}

function countContact(tally, observed, expected) {
  if (observed.first !== null && expected.first !== null) {
    tally.tp++;
    tally.onsetFrames += Math.abs(observed.first - expected.first);
    tally.releaseFrames += Math.abs(observed.last - expected.last);
  } else if (observed.first !== null) tally.fp++;
  else if (expected.first !== null) tally.fn++;
  else tally.tn++;
}

const metrics = Object.fromEntries(references.map((key) =>
  [key, Object.fromEntries(variants.map((variant) => [variant, makeMetrics()]))]));
let fallbackIntervals = 0;
let evaluatedJoins = 0;
let evaluatedClosures = 0;
let affectedClosures = 0;
const closureChanges = [];
const clipReports = [];
const selectionMs = Object.fromEntries(variants.map((variant) => [variant, 0]));
const secondsAt = (frame) => frame / FPS;
const overlap = (from, to, spans) => spans.some((span) => span.start < to && from < span.end);

for (const row of rows) {
  const [pcm, weights, geometry] = await Promise.all(['pcm', 'arkit', 'geometry-gaps'].map((ext) =>
    floats(new URL(row.id + '.' + ext, local))));
  const plans = variants.map((variant) => {
    const start = performance.now();
    const plan = planAnimation(dictionary, row.phones, pcm, {
      singleFallback: joinComparison || variant === 'with',
      joinAware: variant === 'joinAware',
    });
    selectionMs[variant] += performance.now() - start;
    return plan;
  });
  const [beforePlan, afterPlan] = plans;
  if (joinComparison) {
    assert.deepEqual(beforePlan.coverage, afterPlan.coverage);
  } else {
    assert.equal(beforePlan.coverage.triple, afterPlan.coverage.triple);
    assert.equal(beforePlan.coverage.pair, afterPlan.coverage.pair);
    assert.equal(beforePlan.coverage.miss, afterPlan.coverage.fallback + afterPlan.coverage.miss);
  }
  const spans = afterPlan.segments.filter(({ kind }) => kind === 'fallback');
  fallbackIntervals += spans.length;
  const tracks = {
    ...Object.fromEntries(variants.map((variant, i) => [variant, gaps(bakeAnimation(dictionary, plans[i]))])),
    geometry, fitted: gaps(weights),
  };
  const frameCount = geometry.length / landmarks;
  for (const track of Object.values(tracks)) {
    assert.equal(track.length, geometry.length);
    assert.ok(track.every(Number.isFinite));
  }
  // Merge shared anchors and exclude clip-edge fades. Each join gets one peak
  // measurement even when its frame window overlaps a nearby join window.
  const anchors = spans.flatMap(({ start, end }) => [start, end]);
  if (joinComparison) {
    // Use baseline anchor positions for both variants, independent of which
    // source sample was chosen. Include the middle of each triple overlap.
    for (const [i, { start, end, kind, pieces }] of beforePlan.segments.entries()) {
      if (kind === 'triple' && pieces.length === 2) anchors.push((start + end) / 2);
      const previous = beforePlan.segments[i - 1];
      const outgoing = previous?.pieces.at(-1);
      const incoming = pieces[0];
      if (previous?.last && beforePlan.segments[i].first && outgoing && incoming &&
          outgoing.match !== incoming.match && previous.kind !== 'fallback' && kind !== 'fallback') {
        anchors.push(start);
      }
    }
  }
  const joins = [...new Set(anchors)].sort((a, b) => a - b);
  evaluatedJoins += joins.length;
  const closureEvents = row.phones.filter(({ phone }) => ['M', 'P', 'B'].includes(phone)).map((phone) => {
    const start = phone.start - padding, end = phone.end + padding;
    const from = Math.max(0, Math.ceil(start * FPS));
    const to = Math.min(frameCount, Math.ceil(end * FPS));
    const observations = Object.fromEntries(Object.entries(tracks)
      .map(([name, track]) => [name, contact(track, from, to)]));
    return { phone, affected: overlap(start, end, spans), observations };
  });
  evaluatedClosures += closureEvents.length;
  affectedClosures += closureEvents.filter(({ affected }) => affected).length;
  for (const { phone, affected, observations } of closureEvents) {
    const before = observations[variants[0]].first !== null;
    const after = observations[variants[1]].first !== null;
    if (before !== after) closureChanges.push({
      text: row.text, phone: phone.phone, start: phone.start, affected,
      geometry: observations.geometry.first !== null, fitted: observations.fitted.first !== null,
      [variants[0]]: before, [variants[1]]: after,
    });
  }

  const clipReport = { text: row.text, fallbackIntervals: spans.length, affectedLipGapRms: {} };
  for (const reference of references) {
    for (const variant of variants) {
      const tally = metrics[reference][variant];
      const actual = tracks[variant], expected = tracks[reference];
      let affectedSquares = 0, affectedValues = 0;
      for (let frame = 0; frame < frameCount; frame++) {
        const time = secondsAt(frame);
        const affected = spans.some(({ start, end }) => start <= time && time < end);
        for (let j = 0; j < landmarks; j++) {
          const i = frame * landmarks + j;
          const squared = (actual[i] - expected[i]) ** 2;
          tally.squares += squared;
          tally.values++;
          if (affected) {
            affectedSquares += squared;
            affectedValues++;
          }
        }
      }
      tally.affectedSquares += affectedSquares;
      tally.affectedValues += affectedValues;
      if (reference === 'geometry' && affectedValues) {
        clipReport.affectedLipGapRms[variant] = Math.sqrt(affectedSquares / affectedValues);
      }
      // Compare the change in gap per output frame, using step midpoints
      // to decide whether a frame pair belongs to a join window.
      const seenSteps = new Set();
      for (const anchor of joins) {
        let actualPeak = 0, expectedPeak = 0;
        for (let frame = 1; frame < frameCount; frame++) {
          if (Math.abs((frame - 0.5) / FPS - anchor) > joinRadius) continue;
          for (let j = 0; j < landmarks; j++) {
            const i = frame * landmarks + j;
            const actualStep = actual[i] - actual[i - landmarks];
            const expectedStep = expected[i] - expected[i - landmarks];
            actualPeak = Math.max(actualPeak, Math.abs(actualStep));
            expectedPeak = Math.max(expectedPeak, Math.abs(expectedStep));
            if (!seenSteps.has(frame)) {
              tally.joinSquares += (actualStep - expectedStep) ** 2;
              tally.joinValues++;
            }
          }
          seenSteps.add(frame);
        }
        tally.peakStep += actualPeak;
        tally.joinCount++;
        if (actualPeak > expectedPeak) tally.joinsOverReference++;
      }
      for (const { affected, observations } of closureEvents) {
        countContact(tally.closure, observations[variant], observations[reference]);
        if (affected) countContact(tally.affectedClosure, observations[variant], observations[reference]);
      }
    }
  }
  if (spans.length) clipReports.push(clipReport);
}

const ratio = (a, b) => b ? a / b : null;
const rms = (a, b) => b ? Math.sqrt(a / b) : null;
function summarizeClosure({ tp, fp, fn, tn, onsetFrames, releaseFrames }) {
  return {
    tp, fp, fn, tn, precision: ratio(tp, tp + fp), recall: ratio(tp, tp + fn),
    f1: ratio(2 * tp, 2 * tp + fp + fn),
    meanOnsetErrorMs: ratio(onsetFrames * 1000, tp * FPS),
    meanReleaseErrorMs: ratio(releaseFrames * 1000, tp * FPS),
  };
}
function summarize(tally) {
  return {
    lipGapRms: rms(tally.squares, tally.values),
    fallbackSpanLipGapRms: rms(tally.affectedSquares, tally.affectedValues),
    joinStepErrorRms: rms(tally.joinSquares, tally.joinValues),
    meanJoinPeakStep: ratio(tally.peakStep, tally.joinCount),
    joinsOverReference: tally.joinsOverReference,
    closures: summarizeClosure(tally.closure),
    closuresNearFallback: summarizeClosure(tally.affectedClosure),
  };
}

const report = {
  sentences: rows.length, fallbackIntervals, evaluatedJoins, evaluatedClosures, affectedClosures,
  meanSelectionMs: Object.fromEntries(variants.map((key) => [key, selectionMs[key] / rows.length])),
  settings: { fps: FPS, landmarks, threshold, padding, joinRadius },
  results: Object.fromEntries(references.map((key) =>
    [key, Object.fromEntries(variants.map((variant) => [variant, summarize(metrics[key][variant])]))])),
  closureChanges, clipReports,
};
await writeFile(new URL(joinComparison ? 'join-quality.json' : 'fallback-quality.json', local), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));

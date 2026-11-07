import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { bakeAnimation, planAnimation } from '../src/lib/dictionary/animate.mjs';
import { CHANNELS, FPS } from '../src/lib/dictionary/index.mjs';
import { checkAdvanceClosures, summarizeAdvanceClosures, validateClosureProfile } from './lib/closure-quality.mjs';

const testSet = process.argv.includes('--test-set');
const developmentSet = process.argv.includes('--development-set');
assert.ok(!(testSet && developmentSet), 'Choose one evaluation split');
const optimizationComparison = process.argv.includes('--optimization');
const visemeComparison = process.argv.includes('--viseme-fallback');
const advanceComparison = process.argv.includes('--advance-closures');
const openingComparison = process.argv.includes('--opening-context');
const boundedComparison = process.argv.includes('--bounded-context');
assert.ok([optimizationComparison, visemeComparison, advanceComparison, openingComparison, boundedComparison].filter(Boolean).length <= 1, 'Compare one feature at a time');
const acousticOnly = process.argv.includes('--acoustic-only');
const retimingOnly = process.argv.includes('--retiming-only');
assert.ok(!(acousticOnly && retimingOnly), 'Choose one ablation');
assert.ok(!(acousticOnly || retimingOnly) || optimizationComparison, 'Ablations require --optimization');
const argument = (name) => {
  const at = process.argv.indexOf(name);
  if (at < 0) return undefined;
  const value = process.argv[at + 1];
  assert.ok(value && !value.startsWith('--'), name + ' requires a path');
  return value;
};
const baselinePath = argument('--baseline');
const outputPath = argument('--output');
const acousticMode = argument('--acoustic-mode') ?? 'predicted';
const contextStrength = Number(argument('--context-strength') ?? '.25');
const contextPhones = argument('--context-phones')?.split(',');
const contextBlend = argument('--context-blend') ?? 'anchored';
assert.ok(['absolute','anchored'].includes(contextBlend), 'Unknown context blend mode');
assert.ok(!contextPhones || boundedComparison, '--context-phones requires --bounded-context');
assert.ok(Number.isFinite(contextStrength) && contextStrength > 0 && contextStrength <= .5, 'Context strength must be in (0, 0.5]');
assert.ok(['predicted', 'similarity'].includes(acousticMode), 'Unknown acoustic mode');
const corpusComparison = Boolean(baselinePath) || optimizationComparison || visemeComparison || advanceComparison || openingComparison || boundedComparison;
const local = new URL(testSet || developmentSet ? '../.local/full/' : '../.local/fresh/', import.meta.url);
const packed = new URL('../static/dictionary/', import.meta.url);
const json = async (url) => JSON.parse(await readFile(url));
const floats = async (url) => {
  const bytes = await readFile(url);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};

const dictionary = await json(new URL('index.json', packed));
const closureProfile = await json(new URL('../data/closure-windows.json', import.meta.url));
validateClosureProfile(closureProfile);
assert.ok(['targeted', 'subtitles'].includes(dictionary.corpus), 'Pack the targeted or subtitles corpus first');
dictionary.weights = await floats(new URL(dictionary.weightsFile, packed));
let baseline;
if (baselinePath) {
  const source = pathToFileURL(resolve(baselinePath) + '/');
  baseline = await json(new URL('index.json', source));
  baseline.weights = await floats(new URL(baseline.weightsFile, source));
  assert.deepEqual(baseline.channels, dictionary.channels);
  assert.equal(baseline.fit, dictionary.fit);
  assert.deepEqual(baseline.lipRig, dictionary.lipRig);
  assert.deepEqual(baseline.closureModel, dictionary.closureModel, 'Keep the closure model fixed for corpus comparison');
}
if (optimizationComparison || visemeComparison || advanceComparison || openingComparison || boundedComparison) {
  if (optimizationComparison && !retimingOnly) assert.ok(dictionary.clips.every((clip) => clip.acoustics) && (acousticMode === 'similarity' || dictionary.selectionModel), 'Pack with --acoustics and prepare the selection model first');
  baseline ??= dictionary;
  assert.deepEqual(baseline.clips.map(({ id }) => id), dictionary.clips.map(({ id }) => id), 'Optimization comparison must keep the corpus fixed');
  assert.equal(baseline.weightsFile, dictionary.weightsFile, 'Optimization comparison must keep the animation tracks fixed');
}
const [allRows, rig] = await Promise.all([
  json(new URL('clips.json', local)),
  json(new URL('lip-rig.json', local)),
]);
const rows = testSet || developmentSet ? allRows.filter(({ split }) => split === (developmentSet ? 'development' : 'test')) : allRows;
const landmarks = rig.neutral.length;
const center = Math.floor(landmarks / 2);
const closureComparison = process.argv.includes('--closures');
const joinComparison = corpusComparison || closureComparison || process.argv.includes('--join-aware');
const variants = boundedComparison ? ['baseline', 'boundedContext'] : openingComparison ? ['baseline', 'openingContext'] : advanceComparison ? ['baseline', 'advanceClosures'] : visemeComparison ? ['baseline', 'visemeFallback'] : optimizationComparison ? ['baseline', 'optimized'] : corpusComparison ? ['baseline', 'expanded'] : closureComparison ? ['current', 'closureAware'] : joinComparison ? ['timingOnly', 'joinAware'] : ['without', 'with'];
const dictionaries = corpusComparison ? [baseline, dictionary] : [dictionary, dictionary];
const references = ['geometry', 'fitted'];
const fitExtension = dictionary.fit === 'lip' ? 'lip.arkit' : 'arkit';
const directFit = makeMetrics();
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
    globalClosure: { tp: 0, fp: 0, fn: 0, tn: 0, onsetFrames: 0, releaseFrames: 0 },
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

function contactEpisodes(track) {
  const episodes = [];
  let active;
  for (let frame = 0; frame < track.length / landmarks; frame++) {
    if (track[frame * landmarks + center] <= threshold) {
      if (!active) { active = { first: frame, last: frame }; episodes.push(active); }
      active.last = frame;
    } else active = undefined;
  }
  return episodes;
}

function countGlobalContacts(tally, actual, reference) {
  const observed = contactEpisodes(actual), expected = contactEpisodes(reference);
  const events = { matched: [], extra: [], missed: [] };
  const absent = { first: null, last: null };
  let a = 0, b = 0;
  while (a < observed.length || b < expected.length) {
    if (b === expected.length || (a < observed.length && observed[a].last + padding * FPS < expected[b].first)) {
      events.extra.push(observed[a]);
      countContact(tally, observed[a++], absent);
    } else if (a === observed.length || expected[b].last + padding * FPS < observed[a].first) {
      events.missed.push(expected[b]);
      countContact(tally, absent, expected[b++]);
    } else {
      events.matched.push({ actual: observed[a], expected: expected[b] });
      countContact(tally, observed[a++], expected[b++]);
    }
  }
  return events;
}

const metrics = Object.fromEntries(references.map((key) =>
  [key, Object.fromEntries(variants.map((variant) => [variant, makeMetrics()]))]));
const advanceChecks = Object.fromEntries(references.map((key) => [key, Object.fromEntries(variants.map((variant) => [variant, []]))]));
const directFitAdvance = [];
const centerTrack = (track) => Float32Array.from({ length: track.length / landmarks }, (_, frame) => track[frame * landmarks + center]);
let fallbackIntervals = 0;
let evaluatedJoins = 0;
let evaluatedClosures = 0;
let affectedClosures = 0;
const closureChanges = [];
const clipReports = [];
const selectionMs = Object.fromEntries(variants.map((variant) => [variant, 0]));
const coverage = Object.fromEntries(variants.map((variant) => [variant, { triple: 0, pair: 0, single: 0, fallback: 0, miss: 0, approximate: 0 }]));
const usage = Object.fromEntries(variants.map((variant) => [variant, { triple: 0, pair: 0, single: 0, approximate: 0 }]));
const approximateMatches = [];
const normalize = (text) => text.toLowerCase().replace(/[^a-z]+/g, ' ').trim();
if (corpusComparison) {
  const trainingTexts = new Set();
  for (const name of ['full', 'targeted', 'subtitles']) {
    const manifest = await json(new URL('../.local/' + name + '/clips.json', import.meta.url));
    for (const row of manifest) if (row.split === 'train') trainingTexts.add(normalize(row.text));
  }
  for (const row of rows) assert.ok(!trainingTexts.has(normalize(row.text)), 'Evaluation transcript is in training: ' + row.text);
}
const secondsAt = (frame) => frame / FPS;
const overlap = (from, to, spans) => spans.some((span) => span.start < to && from < span.end);

for (const row of rows) {
  const [pcm, weights, geometry] = await Promise.all(['pcm', fitExtension, 'geometry-gaps'].map((ext) =>
    floats(new URL(row.id + '.' + ext, local))));
  const plans = variants.map((variant, i) => {
    const start = performance.now();
    const plan = planAnimation(dictionaries[i], row.phones, pcm, {
      singleFallback: joinComparison || variant === 'with',
      joinAware: corpusComparison || closureComparison || variant === 'joinAware',
      closureAware: corpusComparison || variant === 'closureAware',
      acousticAware: optimizationComparison && variant === 'optimized' && !retimingOnly,
      closureRetiming: optimizationComparison && variant === 'optimized' && !acousticOnly,
      acousticMode,
      visemeFallback: boundedComparison || openingComparison || advanceComparison || (visemeComparison && variant === 'visemeFallback'),
      advanceClosures: boundedComparison || openingComparison || (advanceComparison && variant === 'advanceClosures'),
      openingContext: boundedComparison || (openingComparison && variant === 'openingContext'),
      boundedContext: boundedComparison && variant === 'boundedContext', contextStrength, contextBlend,
      smoothNeutral: false, // Historical comparisons retain the original neutral transition.
    });
    if (contextPhones) plan.contexts = plan.contexts.filter(context => contextPhones.includes(context.phone));
    selectionMs[variant] += performance.now() - start;
    for (const kind of Object.keys(coverage[variant])) coverage[variant][kind] += plan.coverage[kind];
    for (const kind of Object.keys(usage[variant])) usage[variant][kind] += plan.usage[kind];
    if ((visemeComparison || advanceComparison) && variant !== 'baseline') approximateMatches.push(...plan.approximateMatches.map((match) => ({ ...match, text: row.text })));
    return plan;
  });
  const [beforePlan, afterPlan] = plans;
  if (optimizationComparison || boundedComparison) assert.deepEqual(beforePlan.coverage, afterPlan.coverage, 'Optimization must not change coverage');
  if (boundedComparison) assert.deepEqual(beforePlan.usage, afterPlan.usage, 'Blending must not change selected match counts');
  if (!corpusComparison && joinComparison) {
    assert.deepEqual(beforePlan.coverage, afterPlan.coverage);
  } else if (!corpusComparison) {
    assert.equal(beforePlan.coverage.triple, afterPlan.coverage.triple);
    assert.equal(beforePlan.coverage.pair, afterPlan.coverage.pair);
    assert.equal(beforePlan.coverage.miss, afterPlan.coverage.fallback + afterPlan.coverage.miss);
  }
  const spans = [...new Map((corpusComparison ? plans.flatMap((plan) => plan.segments) : afterPlan.segments)
    .filter(({ kind }) => kind === 'fallback').map((span) => [span.start + ':' + span.end, span])).values()];
  fallbackIntervals += spans.length;
  const tracks = {
    ...Object.fromEntries(variants.map((variant, i) => [variant, gaps(bakeAnimation(dictionaries[i], plans[i]))])),
    geometry, fitted: gaps(weights),
  };
  const frameCount = geometry.length / landmarks;
  for (const track of Object.values(tracks)) {
    assert.equal(track.length, geometry.length);
    assert.ok(track.every(Number.isFinite));
  }
  if (boundedComparison) {
    for (let frame = 0; frame < frameCount; frame++) assert.equal(tracks.boundedContext[frame * landmarks + center] <= threshold,
      tracks.baseline[frame * landmarks + center] <= threshold, 'Context changed the contact mask: ' + row.text);
  }
  for (let i = 0; i < geometry.length; i++) {
    directFit.squares += (tracks.fitted[i] - geometry[i]) ** 2;
    directFit.values++;
  }
  countGlobalContacts(directFit.globalClosure, tracks.fitted, geometry);
  directFitAdvance.push(...checkAdvanceClosures(row.phones, centerTrack(tracks.fitted), centerTrack(geometry), closureProfile));
  // Merge shared anchors and exclude clip-edge fades. Each join gets one peak
  // measurement even when its frame window overlaps a nearby join window.
  const anchors = spans.flatMap(({ start, end }) => [start, end]);
  if (joinComparison) {
    // Both variants use identical measurement windows. Corpus comparison uses
    // the union of their joins; algorithm comparisons use the baseline joins.
    // Include the middle of each triple overlap.
    for (const plan of corpusComparison ? plans : [beforePlan]) {
      for (const [i, { start, end, kind, pieces }] of plan.segments.entries()) {
        if ((kind === 'triple' || kind === 'approximate') && pieces.length === 2) anchors.push((start + end) / 2);
        const previous = plan.segments[i - 1];
        const outgoing = previous?.pieces.at(-1);
        const incoming = pieces[0];
        if (previous?.last && plan.segments[i].first && outgoing && incoming &&
            outgoing.match !== incoming.match && previous.kind !== 'fallback' && kind !== 'fallback') {
          anchors.push(start);
        }
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
    countContact(directFit.closure, observations.fitted, observations.geometry);
    const before = observations[variants[0]].first !== null;
    const after = observations[variants[1]].first !== null;
    if (before !== after) closureChanges.push({
      text: row.text, phone: phone.phone, start: phone.start, affected,
      geometry: observations.geometry.first !== null, fitted: observations.fitted.first !== null,
      [variants[0]]: before, [variants[1]]: after,
    });
  }

  const clipReport = { id: row.id, text: row.text, group: row.split, coverage: Object.fromEntries(variants.map((variant, i) => [variant, plans[i].coverage])), fallbackIntervals: spans.length, affectedLipGapRms: {}, lipGapRms: {}, wholeSpeechContacts: {},
    ...(boundedComparison ? { context: plans[1].contextDiagnostics,
      contextRegions: plans[1].contexts.map((c) => ({ phone: c.phone, side: c.side, start: c.start, end: c.end, sourceClip: c.match.clip.id,
        sourceStart: c.sourceAnchor + (c.start - c.targetAnchor) * c.slope, sourceEnd: c.sourceAnchor + (c.end - c.targetAnchor) * c.slope, strength: c.strength,
        attribution: c.attribution, targetNeighbor: c.targetNeighbor, sourceNeighbor: c.sourceNeighbor })) } : {}),
    ...(openingComparison ? { openingContext: plans[1].leadingContext ? { sourceClip: plans[1].leadingContext.match.clip.id, sourceFrom: plans[1].leadingContext.sourceFrom, sourceTo: plans[1].leadingContext.sourceTo, targetEnd: plans[1].leadingContext.targetEnd } : null } : {}) };
  for (const reference of references) {
    for (const variant of variants) {
      const tally = metrics[reference][variant];
      const actual = tracks[variant], expected = tracks[reference];
      let clipSquares = 0;
      let affectedSquares = 0, affectedValues = 0;
      for (let frame = 0; frame < frameCount; frame++) {
        const time = secondsAt(frame);
        const affected = spans.some(({ start, end }) => start <= time && time < end);
        for (let j = 0; j < landmarks; j++) {
          const i = frame * landmarks + j;
          const squared = (actual[i] - expected[i]) ** 2;
          tally.squares += squared;
          clipSquares += squared;
          tally.values++;
          if (affected) {
            affectedSquares += squared;
            affectedValues++;
          }
        }
      }
      tally.affectedSquares += affectedSquares;
      if (reference === 'geometry') clipReport.lipGapRms[variant] = Math.sqrt(clipSquares / actual.length);
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
      const contacts = countGlobalContacts(tally.globalClosure, actual, expected);
      if (reference === 'geometry') clipReport.wholeSpeechContacts[variant] = contacts;
      advanceChecks[reference][variant].push(...checkAdvanceClosures(row.phones, centerTrack(actual), centerTrack(expected), closureProfile)
        .map((check) => ({ id: row.id, text: row.text, ...check })));
    }
  }
  if (corpusComparison || spans.length) clipReports.push(clipReport);
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
function summarize(tally, advanceRows) {
  return {
    lipGapRms: rms(tally.squares, tally.values),
    fallbackSpanLipGapRms: rms(tally.affectedSquares, tally.affectedValues),
    joinStepErrorRms: rms(tally.joinSquares, tally.joinValues),
    meanJoinPeakStep: ratio(tally.peakStep, tally.joinCount),
    joinsOverReference: tally.joinsOverReference,
    advanceWindowClosures: summarizeAdvanceClosures(advanceRows),
    closures: summarizeClosure(tally.closure),
    closuresNearFallback: summarizeClosure(tally.affectedClosure),
    closuresAcrossSpeech: summarizeClosure(tally.globalClosure),
  };
}

const report = {
  fit: dictionary.fit ?? 'baseline',
  ...(corpusComparison ? { dictionaries: Object.fromEntries(variants.map((variant, i) => [variant, {
    corpus: dictionaries[i].corpus, clips: dictionaries[i].clips.length,
    weightsFile: dictionaries[i].weightsFile,
    uniquePairs: Object.keys(dictionaries[i].index).filter((key) => key.split(' ').length === 2).length,
  }])) } : {}),
  coverage, usage, approximateMatches,
  directFitVsGeometry: {
    lipGapRms: rms(directFit.squares, directFit.values),
    closures: summarizeClosure(directFit.closure),
    closuresAcrossSpeech: summarizeClosure(directFit.globalClosure),
    advanceWindowClosures: summarizeAdvanceClosures(directFitAdvance),
  },
  sentences: rows.length, fallbackIntervals, evaluatedJoins, evaluatedClosures, affectedClosures,
  meanSelectionMs: Object.fromEntries(variants.map((key) => [key, selectionMs[key] / rows.length])),
  settings: { fps: FPS, landmarks, threshold, padding, joinRadius,
    primaryClosureMetric: advanceComparison || openingComparison || boundedComparison ? 'closuresAcrossSpeech' : 'advanceWindowClosures', legacyClosureMetric: 'closures uses the former +/-20ms phoneme window',
    closureProfile,
    ...(boundedComparison ? { boundedContext: true, contextStrength, contactMaskPreserved: true,
      contextPhones: contextPhones ?? 'all', contextBlend,
      contextProfile: await json(new URL('../data/phoneme-context-candidates.json', import.meta.url)),
      bounds: 'Custom phoneme margins capped by one neighboring interval, source/target speech runs and available donor frames; source slope 0.5 to 2; sine-squared taper; feature-specific channels; per-channel contribution capped at 0.5',
      baselineOpeningContext: true, baselineAdvanceClosures: true, baselinePBSubstitution: true } : {}),
    ...(openingComparison ? { openingContext: true, baselineAdvanceClosures: true, baselinePBSubstitution: true, maximumSourceLeadSeconds: .2 } : {}),
    ...(advanceComparison ? { advanceClosureGuidance: true, baselinePBSubstitution: true,
      baselineContactRetention: true,
      selectionAndBlending: 'Advance-window acoustic model, trained on 504 training clips only; recorded contact required for closing blend; high-confidence pose restoration; local retention of baseline contacts' } : {}),
    ...(visemeComparison ? { substitution: 'One P/B swap in a missing triple; exact doubles compete with approximate triples', substitutionsSymmetric: true, acousticAware: false, closureRetiming: false } : {}),
    ...(optimizationComparison ? { acousticMode, acousticWeight: 0.25, optimizedAcoustics: !retimingOnly, optimizedRetiming: !acousticOnly, defaultPlaybackUnchanged: true } : {}),
    ...(corpusComparison ? { singleFallback: true, joinAware: true, closureAware: true, joinWindows: 'union of baseline and expanded anchors' } : {}) },
  results: Object.fromEntries(references.map((key) =>
    [key, Object.fromEntries(variants.map((variant) => [variant, summarize(metrics[key][variant], advanceChecks[key][variant])]))])),
  advanceClosureDetails: advanceChecks.geometry,
  closureChanges, clipReports,
};
const reportFile = outputPath ? pathToFileURL(resolve(outputPath)) : new URL(`${boundedComparison ? 'bounded-context' : openingComparison ? 'opening-context' : advanceComparison ? 'advance-motion' : visemeComparison ? 'viseme' : optimizationComparison ? 'optimization' : corpusComparison ? 'subtitle' : closureComparison ? 'closure' : joinComparison ? 'join' : 'fallback'}-quality${dictionary.fit === 'lip' ? '-lip' : ''}.json`, local);
await writeFile(reportFile, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));

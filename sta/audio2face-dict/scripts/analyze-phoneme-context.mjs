import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { contactEvents, associateFeatureEvents, offsetMeasurements, distribution } from './lib/closure-events.mjs';

const json = async (url) => JSON.parse(await readFile(url));
const floats = async (url) => { const b = await readFile(url); return new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)); };
const root = new URL('../', import.meta.url);
const pointer = await json(new URL('.local/motion-features/current.json', root));
assert.equal(pointer.extractorSHA256, createHash('sha256').update(await readFile(new URL('scripts/lib/motion_features.py', root))).digest('hex'), 'Motion cache uses an older measurement definition; rerun extraction');
const cache = new URL('.local/motion-features/' + pointer.directory + '/', root);
const output = new URL('.local/phoneme-context/', root);
await mkdir(output, { recursive: true });
const clips = [], pooled = Array.from({ length: 4 }, () => []), phoneCounts = {};
const trainingHash = createHash('sha256');
for (const name of ['full','targeted','subtitles']) {
  for (const row of await json(new URL('.local/' + name + '/clips.json', root))) {
    if (row.split !== 'train') continue;
    const [native, fitted] = await Promise.all(['native','fitted'].map((ref) => floats(new URL(row.id + '.' + ref + '.f32', cache))));
    const count = Math.ceil(row.samples * 60 / 16000);
    assert.equal(native.length, count * 4); assert.equal(fitted.length, native.length);
    assert.ok(native.every(Number.isFinite) && fitted.every(Number.isFinite));
    clips.push({ ...row, corpus: name, native, fitted });
    trainingHash.update(JSON.stringify({ id: row.id, samples: row.samples, phones: row.phones }));
    for (const phone of row.phones) phoneCounts[phone.phone] = (phoneCounts[phone.phone] ?? 0) + 1;
    // Threshold calibration excludes distant pre/post silence, but includes
    // preparation/recovery around the utterance. All clips are training only.
    const from = Math.max(0, Math.ceil((row.phones[0].start - .2) * 60));
    const to = Math.min(count, Math.ceil((row.phones.at(-1).end + .2) * 60));
    for (let frame = from; frame < to; frame++) for (let feature = 0; feature < 4; feature++) pooled[feature].push(native[frame * 4 + feature]);
  }
}
assert.equal(clips.length, pointer.trainingClips);
assert.equal(new Set(clips.map((c) => c.id)).size, clips.length);
const closureProfile = await json(new URL('data/closure-windows.json', root));
assert.equal(trainingHash.digest('hex'), closureProfile.trainingSetSha256, 'Existing closure profile and motion features use different training manifests');
const quantile = (values, fraction) => {
  const ordered = [...values].sort((a,b) => a-b), at = fraction * (ordered.length - 1), lo = Math.floor(at);
  return ordered[lo] + (ordered[Math.min(lo+1,ordered.length-1)] - ordered[lo]) * (at-lo);
};
const groups = [
  { name: 'rounding', labels: ['W','UW','OW','UH','AO','OY','R'], features: [0,1], definition: 'Both mouth narrowing and lip protrusion exceed training feature thresholds; geometric rounding proxy. R is included as an ownership probe so its nearby motion is not silently credited to a rounded vowel.' },
  { name: 'jawOpening', labels: 'AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW'.split(' '), features: [2], definition: 'Mean downward jaw marker displacement above training threshold' },
  { name: 'fvProximity', labels: ['F','V'], features: [3], definition: 'Lower-lip proximity to upper incisor vertices above training threshold; not confirmed physical contact' },
];
const configs = [.65,.75,.85].flatMap((fraction) => [.15,.2,.25].map((window) => ({ fraction, window, records: [] })));
const main = configs.find((c) => c.fraction === .75 && c.window === .2);
for (const config of configs) {
  const thresholds = pooled.map((values) => quantile(values, config.fraction));
  for (const clip of clips) for (const group of groups) for (const reference of ['native','fitted']) {
    const track = clip[reference], count = track.length / 4;
    // Positive margin means every required geometric feature is active.
    const margin = Float32Array.from({ length: count }, (_, frame) => Math.min(...group.features.map((feature) =>
      (track[frame*4+feature] - thresholds[feature]) / Math.max(.001, quantileCache(feature, .75) - quantileCache(feature, .5)))));
    const events = contactEvents(Float32Array.from(margin, (value) => -value), 60, 0);
    for (const a of associateFeatureEvents(clip.phones, events, group.labels, config.window)) {
      const offsets = a.event ? offsetMeasurements(a.phone, a.event) : null;
      if (offsets) {
        offsets.featureDurationMs = offsets.contactDurationMs; delete offsets.contactDurationMs;
        offsets.peakFromPhoneMidpointMs = offsets.deepestFromPhoneMidpointMs; delete offsets.deepestFromPhoneMidpointMs;
      }
      config.records.push({ group: group.name, reference, corpus: clip.corpus, clip: clip.id, phone: a.phone.phone,
        start: a.phone.start, end: a.phone.end, status: a.status,
        offsets,
        event: a.event ? { start: a.event.start, end: a.event.end, peakTime: a.event.deepest, peakMargin: -a.event.minimumGap,
          clippedStart: a.event.clippedStart, clippedEnd: a.event.clippedEnd } : null });
    }
  }
}

function summarize(records) {
  const result = {};
  for (const group of groups) {
    result[group.name] = {};
    for (const phone of group.labels) {
      result[group.name][phone] = {};
      for (const ref of ['native','fitted']) {
        const rows = records.filter((r) => r.group === group.name && r.phone === phone && r.reference === ref);
        const counts = {}, matched = rows.filter((r) => r.offsets);
        for (const r of rows) counts[r.status] = (counts[r.status] ?? 0) + 1;
        result[group.name][phone][ref] = { total: rows.length, matched: matched.length, counts,
          offsets: Object.fromEntries(['onsetFromStartMs','releaseFromEndMs','midpointFromPhoneMidpointMs','peakFromPhoneMidpointMs','featureDurationMs'].map((name) =>
            [name, distribution(matched.map((r) => r.offsets[name]))])) };
      }
    }
  }
  return result;
}
const mainSummary = summarize(main.records);
const sensitivities = configs.filter((c) => c !== main).map((c) => ({ thresholdQuantile: c.fraction, associationWindowMs: c.window * 1000, summary: summarize(c.records) }));
const candidates = {};
for (const phone of Object.keys(phoneCounts).sort()) {
  const evidence = [];
  const closure = closureProfile.phones[phone];
  if (closure) evidence.push({ feature: 'lipClosure', examples: closure.examples,
    beforeMs: Math.ceil((Math.max(0,-closure.onsetRangeMs[0]) + 1000/60) / (1000/60)) * 1000/60,
    afterMs: Math.ceil((Math.max(0,closure.releaseRangeMs[1]) + 1000/60) / (1000/60)) * 1000/60,
    source: 'data/closure-windows.json', onsetMedianSensitivitySpreadMs: null });
  for (const group of groups.filter((group) => group.labels.includes(phone))) {
    const value = mainSummary[group.name][phone].native;
    if (value.matched < 30) continue;
    const variants = [value, ...sensitivities.filter((s) => s.associationWindowMs === 200).map((s) => s.summary[group.name][phone].native)].filter((x) => x.matched >= 20);
    const before = Math.max(0, -Math.min(...variants.map((v) => v.offsets.onsetFromStartMs.p05)));
    const after = Math.max(0, Math.max(...variants.map((v) => v.offsets.releaseFromEndMs.p95)));
    evidence.push({ feature: group.name, examples: value.matched, matchedFraction: value.matched / value.total,
      beforeMs: Math.ceil((before + 1000/60) / (1000/60)) * 1000/60,
      afterMs: Math.ceil((after + 1000/60) / (1000/60)) * 1000/60,
      onsetMedianSensitivitySpreadMs: Math.max(...variants.map((v) => v.offsets.onsetFromStartMs.median)) - Math.min(...variants.map((v) => v.offsets.onsetFromStartMs.median)) });
  }
  candidates[phone] = { occurrences: phoneCounts[phone], active: false, evidence,
    candidateBeforeMs: evidence.length ? Math.max(...evidence.map((v) => v.beforeMs)) : null,
    candidateAfterMs: evidence.length ? Math.max(...evidence.map((v) => v.afterMs)) : null,
    thresholdSensitive: evidence.some((v) => v.onsetMedianSensitivitySpreadMs > 1000 / 30),
    broadContext: evidence.some((v) => v.beforeMs > 250 || v.afterMs > 250),
    status: evidence.length ? 'candidate_requires_playback_validation' : 'insufficient_unambiguous_feature_evidence' };
}
const report = { training: pointer, method: { mainThresholdQuantile: .75, associationWindowMs: 200, featureGroups: groups,
  thresholds: Object.fromEntries(pointer.features.map((name,i) => [name, quantileCache(i,.75)])),
  association: 'Only a unique complete event with no other eligible phoneme claiming it enters primary timing statistics; shared, absent, clipped and multiple events are counted separately.',
  candidateWindow: 'Envelope of P05 onset / P95 release across 65th,75th,85th percentile feature thresholds, plus one 60 fps frame on each side; minimum 30 unambiguous main examples. These are proposals, not playback settings.',
  caveats: ['Continuous geometric features have no unique physical onset; threshold sensitivity is reported.', 'F/V measurement is proximity to dense incisor vertices, not proven surface contact.', 'Shared coarticulation can prevent reliable attribution to individual phonemes.', 'No evaluation/development clips used; pipeline-specific offsets include model/alignment timing.'] },
  summary: mainSummary, sensitivity: sensitivities, candidates };
await writeFile(new URL('summary.json', output), JSON.stringify(report,null,2)+'\n');
await writeFile(new URL('events.json', output), JSON.stringify(main.records,null,2)+'\n');
const candidateFile = { version: 1, trainingManifestSHA256: pointer.trainingManifestSHA256,
  closureTrainingSetSHA256: closureProfile.trainingSetSha256, active: false,
  description: 'Provisional feature-context margins, not validated playback settings. Broad ranges can contain unrelated neighbor motion and must be bounded by donor events and target context.', phonemes: candidates };
await writeFile(new URL('candidates.json', output), JSON.stringify(candidateFile,null,2)+'\n');
if (process.argv.includes('--write-candidates')) await writeFile(new URL('data/phoneme-context-candidates.json', root), JSON.stringify(candidateFile,null,2)+'\n');
const csv = [['phoneme','feature','reference','occurrences','matched_examples','onset_median_ms','onset_p05_ms','onset_p95_ms','release_median_ms','release_p05_ms','release_p95_ms']];
for (const group of groups) for (const phone of group.labels) for (const ref of ['native','fitted']) {
  const row = mainSummary[group.name][phone][ref], a = row.offsets.onsetFromStartMs, b = row.offsets.releaseFromEndMs;
  csv.push([phone,group.name,ref,row.total,row.matched,...['median','p05','p95'].map((k) => a[k] ?? ''),...['median','p05','p95'].map((k) => b[k] ?? '')]);
}
await writeFile(new URL('timing.csv', output), csv.map((row) => row.join(',')).join('\n')+'\n');
console.log(JSON.stringify({ clips: clips.length, candidates },null,2));

// Memoized population quantiles avoid sorting inside feature/frame loops.
function quantileCache(feature, fraction) {
  const key = feature + ':' + fraction;
  quantileCache.values ??= new Map();
  if (!quantileCache.values.has(key)) quantileCache.values.set(key, quantile(pooled[feature], fraction));
  return quantileCache.values.get(key);
}

import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { contactEvents, associateContacts, offsetMeasurements, distribution } from './lib/closure-events.mjs';
import { makeClosureProfile } from './lib/closure-quality.mjs';

const json = async (url) => JSON.parse(await readFile(url));
const floats = async (url) => {
  const bytes = await readFile(url);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const root = new URL('../.local/', import.meta.url);
const output = new URL('closure-offsets/', root);
await mkdir(output, { recursive: true });
const configs = [0.05, 0.1, 0.15].flatMap((threshold) => [0.15, 0.2, 0.25].map((window) => ({ threshold, window, records: [] })));
const main = configs.find(({ threshold, window }) => threshold === 0.1 && window === 0.2);
const clipIds = new Set();
const trainingHash = createHash('sha256');
const sources = {};
let seconds = 0;
for (const name of ['full', 'targeted', 'subtitles']) {
  const directory = new URL(name + '/', root);
  const manifest = await json(new URL('clips.json', directory));
  const rig = await json(new URL('lip-rig.json', directory));
  const rows = manifest.filter(({ split }) => split === 'train');
  sources[name] = rows.length;
  for (const row of rows) {
    assert.ok(!clipIds.has(row.id), 'Duplicate training clip ' + row.id);
    clipIds.add(row.id);
    trainingHash.update(JSON.stringify({ id: row.id, samples: row.samples, phones: row.phones }));
    seconds += row.samples / 16000;
    const count = Math.ceil(row.samples * 60 / 16000);
    const [native, weights] = await Promise.all(['geometry-gaps', 'lip.arkit'].map((extension) => floats(new URL(row.id + '.' + extension, directory))));
    assert.equal(native.length, count * rig.neutral.length, 'Incomplete native geometry reference for ' + row.id);
    assert.equal(weights.length, count * 52, 'Incomplete ARKit fit for ' + row.id);
    const center = Math.floor(rig.neutral.length / 2);
    const geometry = new Float32Array(count), fitted = new Float32Array(count);
    for (let frame = 0; frame < count; frame++) {
      geometry[frame] = native[frame * rig.neutral.length + center];
      let gap = rig.neutral[center];
      for (let channel = 0; channel < 52; channel++) gap += rig.deltas[channel][center] * weights[frame * 52 + channel];
      fitted[frame] = gap;
    }
    for (const config of configs) {
      const a = associateContacts(row.phones, contactEvents(geometry, 60, config.threshold), config.window);
      const b = associateContacts(row.phones, contactEvents(fitted, 60, config.threshold), config.window);
      assert.equal(a.length, b.length);
      for (let i = 0; i < a.length; i++) {
        const describe = (association) => ({ status: association.status, candidates: association.candidates,
          event: association.event, offsets: association.event ? offsetMeasurements(association.phone, association.event) : null,
          nearestCandidate: association.nearestCandidate });
        config.records.push({ corpus: name, clip: row.id, text: row.text, phone: a[i].phone.phone,
          phoneIndex: a[i].phone.index, start: a[i].phone.start, end: a[i].phone.end,
          geometry: describe(a[i]), fitted: describe(b[i]) });
      }
    }
  }
}
const measurements = ['onsetFromStartMs', 'releaseFromEndMs', 'midpointFromPhoneMidpointMs', 'deepestFromPhoneMidpointMs', 'contactDurationMs'];
function summarize(records) {
  return Object.fromEntries(['P', 'B', 'M'].map((phone) => {
    const rows = records.filter((row) => row.phone === phone);
    const references = Object.fromEntries(['geometry', 'fitted'].map((reference) => {
      const counts = {};
      for (const row of rows) counts[row[reference].status] = (counts[row[reference].status] ?? 0) + 1;
      const matched = rows.filter((row) => row[reference].offsets);
      return [reference, { associationCounts: counts, matched: matched.length,
        offsets: Object.fromEntries(measurements.map((name) => [name, distribution(matched.map((row) => row[reference].offsets[name]))])),
        contactEntirelyBefore20msWindow: matched.filter((row) => row[reference].event.end < row.start - .02).length,
        onsetBeforePhone: matched.filter((row) => row[reference].event.start < row.start).length }];
    }));
    const paired = rows.filter((row) => row.geometry.offsets && row.fitted.offsets);
    return [phone, { total: rows.length, ...references, pairedFitMinusGeometry: {
      n: paired.length, onsetMs: distribution(paired.map((row) => row.fitted.offsets.onsetFromStartMs - row.geometry.offsets.onsetFromStartMs)),
      releaseMs: distribution(paired.map((row) => row.fitted.offsets.releaseFromEndMs - row.geometry.offsets.releaseFromEndMs)) } }];
  }));
}
const report = { trainingClips: clipIds.size, trainingSeconds: seconds, trainingSetSha256: trainingHash.digest('hex'), sources,
  method: { fps: 60, centerLipContactThreshold: main.threshold, associationPaddingMs: main.window * 1000,
    association: 'Primary statistics require exactly one contact episode near the phoneme and no other P/B/M eligible for that episode. Ambiguous/shared candidates, absent contacts and clip-truncated episodes are reported separately.',
    timing: 'Threshold crossings interpolated between samples; underlying resolution remains 16.7 ms.',
    offsets: 'Negative onset means contact before acoustic phoneme start; negative release means reopening before acoustic phoneme end.',
    population: 'Full native Audio2Face v3.2 James outputs and their existing ARKit lip fits on Phonon Easton training speech, aligned by full TinyAligner. Evaluation and development clips excluded.' },
  summary: summarize(main.records),
  byCorpus: Object.fromEntries(Object.keys(sources).map((name) => [name, summarize(main.records.filter((row) => row.corpus === name))])),
  sensitivity: configs.filter((config) => config !== main).map((config) => ({ threshold: config.threshold, associationPaddingMs: config.window * 1000, summary: summarize(config.records) })),
};
await writeFile(new URL('summary.json', output), JSON.stringify(report, null, 2) + '\n');
if (process.argv.includes('--write-profile')) {
  const directory = new URL('../data/', import.meta.url);
  await mkdir(directory, { recursive: true });
  await writeFile(new URL('closure-windows.json', directory), JSON.stringify(makeClosureProfile(report), null, 2) + '\n');
}
await writeFile(new URL('associations.json', output), JSON.stringify(main.records, null, 2) + '\n');
const csv = [['corpus','clip','phoneme','start_s','end_s','reference','association','onset_offset_ms','release_offset_ms','center_offset_ms','deepest_offset_ms','contact_duration_ms']];
for (const row of main.records) for (const reference of ['geometry','fitted']) {
  const value = row[reference];
  csv.push([row.corpus,row.clip,row.phone,row.start,row.end,reference,value.status,...measurements.map((name) => value.offsets?.[name] ?? '')]);
}
await writeFile(new URL('offsets.csv', output), csv.map((row) => row.join(',')).join('\n') + '\n');
console.log(JSON.stringify({ trainingClips: report.trainingClips, trainingSeconds: seconds, summary: report.summary }, null, 2));

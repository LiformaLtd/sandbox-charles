import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { bakeAnimation, joinSourceTimes, planAnimation } from '../src/lib/dictionary/animate.mjs';

const pilot = new URL('../.local/pilot/', import.meta.url);
const dictionary = new URL('../static/dictionary/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('index.json', dictionary)));
assert.equal(manifest.corpus, 'pilot');
const readFloats = async (url) => new Float32Array(await readFile(url).then((bytes) =>
  bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
manifest.weights = await readFloats(new URL(manifest.weightsFile, dictionary));
const rows = JSON.parse(await readFile(new URL('clips.json', pilot)));
const cases = new Map();
for (const row of rows.filter(({ split }) => split === 'development')) {
  const pcm = await readFloats(new URL(row.id + '.pcm', pilot));
  const plan = planAnimation(manifest, row.phones, pcm);
  const output = bakeAnimation(manifest, plan);
  assert.ok(output.every(Number.isFinite));
  console.log(row.text, plan.coverage, `${plan.joins.length} joins`);
  for (const join of plan.joins) {
    const type = [join.outgoing.match.count, join.incoming.match.count].sort().join('-');
    if (join.start >= join.anchor || join.end <= join.anchor) continue;
    const [, incoming] = joinSourceTimes(join, (join.start + join.anchor) / 2);
    const [outgoing] = joinSourceTimes(join, (join.anchor + join.end) / 2);
    assert.ok(incoming < join.incoming.from && outgoing > join.outgoing.to);
    if (!cases.has(type)) cases.set(type, []);
    cases.get(type).push({ phone: join.phone.phone, start: join.start, end: join.end });
  }
}
assert.ok(cases.has('2-2'), 'Pilot must exercise a real double-to-double join');
assert.ok(cases.has('2-3'), 'Pilot must exercise a real mixed join');
console.log('Verified extended native source tracks:', Object.fromEntries([...cases].map(([type, joins]) => [type, joins.length])));

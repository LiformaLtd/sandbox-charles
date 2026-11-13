import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';

const base = process.env.TEST_BASE_URL;
async function post(body, origin = base) {
	return fetch(`${base}/api/blends`, { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: origin }, body: JSON.stringify(body) });
}
const recipe = { sourceA: 'Alfie', sourceB: 'Hugh', percentB: 50, action: 'preview' };
function values(bytes, generated = true) {
	const b = Buffer.from(bytes), n = Number(b.readBigUInt64LE());
	const header = JSON.parse(b.subarray(8, 8 + n).toString());
	assert.deepEqual(header.speaker_wavs.shape, [1, 512, 125]);
	assert.equal(header.speaker_wavs.dtype, 'F32');
	if (generated) assert.deepEqual(header.__metadata__, { sig: 'ba7a5a45', epoch: '300' });
	return Array.from({ length: 512 * 125 }, (_, i) => b.readFloatLE(8 + n + i * 4));
}

test('blend arithmetic, persistence and safe input handling', { skip: !base, timeout: 180_000 }, async () => {
	const initial = await (await fetch(`${base}/api/blends`)).json();
	for (const invalid of [{ percentB: -1 }, { percentB: 101 }, { percentB: 1.5 }, { sourceA: '../file' }, { sourceB: 'https://example.com' }]) {
		assert.equal((await post({ ...recipe, ...invalid })).status, 400);
	}
	assert.equal((await post(recipe, 'https://example.com')).status, 403);
	assert.equal((await fetch(`${base}/api/blends/not-an-id`)).status, 404);
	const vectors = [];
	for (const percentB of [0, 50, 100]) {
		const response = await post({ ...recipe, percentB });
		assert.equal(response.status, 200, await response.clone().text());
		vectors.push(values(await response.arrayBuffer()));
	}
	assert.ok(vectors[0].some((v, i) => v !== vectors[2][i]), 'The two source voices should differ');
	const catalogue = JSON.parse(await readFile(new URL('../src/lib/voices-catalogue.json', import.meta.url), 'utf8'));
	for (const [name, endpoint] of [['Alfie', 0], ['Hugh', 2]]) {
		const source = Object.values(catalogue).flat().find(v => v.name === name);
		const bytes = await readFile(resolve(process.env.PHONON_BLEND_DIR || '.data/voice-blends', 'catalogue', basename(source.file)));
		assert.deepEqual(vectors[endpoint], values(bytes, false));
	}
	for (let i = 0; i < vectors[0].length; i++) assert.equal(vectors[1][i], Math.fround((vectors[0][i] + vectors[2][i]) / 2));
	assert.equal((await (await fetch(`${base}/api/blends`)).json()).length, initial.length, 'Previews must not save voices');
	const response = await post({ ...recipe, action: 'save', name: 'QA blend persistence' });
	assert.equal(response.status, 201);
	const voice = await response.json();
	try {
		const listed = await (await fetch(`${base}/api/blends`)).json();
		assert.equal(listed.length, initial.length + 1);
		assert.equal(listed.find(v => v.id === voice.id)?.name, 'QA blend persistence');
		const file = await fetch(`${base}/api/blends/${voice.id}`);
		assert.equal(file.status, 200);
		assert.deepEqual(values(await file.arrayBuffer()), vectors[1]);
		assert.equal((await fetch(`${base}/api/blends/${voice.id}`, { method: 'DELETE', headers: { Origin: 'https://example.com' } })).status, 403);
	} finally {
		assert.equal((await fetch(`${base}/api/blends/${voice.id}`, { method: 'DELETE', headers: { Origin: base } })).status, 200);
	}
	assert.equal((await fetch(`${base}/api/blends/${voice.id}`)).status, 404);
});

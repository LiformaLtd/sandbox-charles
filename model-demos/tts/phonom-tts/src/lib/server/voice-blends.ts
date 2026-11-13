import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { env } from 'node:process';
import catalogue from '../voices-catalogue.json';
import averages from '../average-voices.json';
import type { BlendRecipe, SavedBlend } from '../voice-blends';

const root = resolve(env.PHONON_BLEND_DIR || '.data/voice-blends');
const revision = averages.revision;
const presets = new Map(Object.values(catalogue).flat().map(v => [v.name, v.file]));
const knownSource = (name: string) => presets.has(name) || averages.groups.some(group => group.name === name);
const uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const count = 512 * 125;
const pending = new Map<string, Promise<Buffer>>();

export function validId(id: string) { return uuid.test(id); }
export function validateRecipe(value: unknown): BlendRecipe {
	if (!value || typeof value !== 'object') throw new Error('Choose two available voices and a blend percentage.');
	const v = value as Record<string, unknown>;
	if (typeof v.sourceA !== 'string' || typeof v.sourceB !== 'string' || !knownSource(v.sourceA) || !knownSource(v.sourceB) ||
		typeof v.percentB !== 'number' || !Number.isInteger(v.percentB) || v.percentB < 0 || v.percentB > 100) {
		throw new Error('Choose two available voices and a blend percentage from 0 to 100.');
	}
	return { sourceA: v.sourceA, sourceB: v.sourceB, percentB: v.percentB };
}

/** Decode only the catalogue's documented F32 [1,512,125] voice format. */
export function decodeVoice(bytes: Buffer): Float32Array {
	if (bytes.length < 8 || bytes.length > 512_000) throw new Error('Invalid voice file size.');
	const size = Number(bytes.readBigUInt64LE());
	if (!Number.isSafeInteger(size) || size < 2 || size > 16_384 || 8 + size > bytes.length) throw new Error('Invalid voice header.');
	const header = JSON.parse(bytes.subarray(8, 8 + size).toString('utf8'));
	const tensor = header.speaker_wavs;
	if (tensor?.dtype !== 'F32' || JSON.stringify(tensor.shape) !== '[1,512,125]' ||
		JSON.stringify(tensor.data_offsets) !== `[0,${count * 4}]` || bytes.length !== 8 + size + count * 4) throw new Error('Unsupported catalogue voice format.');
	const values = new Float32Array(count);
	for (let i = 0; i < count; i++) {
		values[i] = bytes.readFloatLE(8 + size + i * 4);
		if (!Number.isFinite(values[i])) throw new Error('Voice file contains invalid values.');
	}
	return values;
}

export function mixVoices(a: Buffer, b: Buffer, percentB: number): Buffer {
	const left = decodeVoice(a), right = decodeVoice(b);
	const weight = percentB / 100;
	const mixed = new Float64Array(count);
	for (let i = 0; i < count; i++) mixed[i] = left[i] * (1 - weight) + right[i] * weight;
	return encodeVoice(mixed);
}

function encodeVoice(values: Float64Array): Buffer {
	const raw = JSON.stringify({ __metadata__: { sig: 'ba7a5a45', epoch: '300' }, speaker_wavs: { dtype: 'F32', shape: [1, 512, 125], data_offsets: [0, count * 4] } });
	const header = Buffer.from(raw.padEnd(Math.ceil(Buffer.byteLength(raw) / 8) * 8, ' '));
	const result = Buffer.alloc(8 + header.length + count * 4);
	result.writeBigUInt64LE(BigInt(header.length)); header.copy(result, 8);
	for (let i = 0; i < count; i++) result.writeFloatLE(values[i], 8 + header.length + i * 4);
	return result;
}

async function catalogueVoice(name: string): Promise<Buffer> {
	const file = presets.get(name)!;
	if (pending.has(file)) return pending.get(file)!;
	const task = (async () => {
		const path = join(root, 'catalogue', file.split('/').pop()!);
		try { return await readFile(path); } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
		const response = await fetch(`https://huggingface.co/gradium/phonon-voices/resolve/${revision}/${file}`, { signal: AbortSignal.timeout(60_000) });
		if (!response.ok) throw new Error(`Could not download ${name} (${response.status}).`);
		const bytes = Buffer.from(await response.arrayBuffer()); decodeVoice(bytes);
		await mkdir(join(root, 'catalogue'), { recursive: true, mode: 0o700 });
		const temporary = `${path}.${randomUUID()}.tmp`;
		await writeFile(temporary, bytes, { mode: 0o600 }); await rename(temporary, path);
		return bytes;
	})();
	pending.set(file, task);
	try { return await task; } finally { pending.delete(file); }
}

export async function createBlend(recipe: BlendRecipe) {
	const [a, b] = await Promise.all([sourceVoice(recipe.sourceA), sourceVoice(recipe.sourceB)]);
	return mixVoices(a, b, recipe.percentB);
}

async function sourceVoice(name: string): Promise<Buffer> {
	const group = averages.groups.find(group => group.name === name);
	return group ? averageVoice(group.id) : catalogueVoice(name);
}

export async function averageVoice(id: string): Promise<Buffer> {
	const group = averages.groups.find(group => group.id === id);
	if (!group) throw new Error('Unknown averaged voice.');
	const fingerprint = createHash('sha256').update(JSON.stringify({ group, revision, formula: 'equal-mean-f32-v1' })).digest('hex').slice(0, 16);
	const folder = join(root, 'averages');
	const path = join(folder, `${id}-${fingerprint}.safetensors`);
	try { const bytes = await readFile(path); decodeVoice(bytes); return bytes; }
	catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
	const sources = await Promise.all(group.members.map(catalogueVoice));
	const mean = new Float64Array(count);
	for (const source of sources) {
		const values = decodeVoice(source);
		for (let i = 0; i < count; i++) mean[i] += values[i];
	}
	for (let i = 0; i < count; i++) mean[i] /= sources.length;
	const bytes = encodeVoice(mean);
	await mkdir(folder, { recursive: true, mode: 0o700 });
	const temporary = `${path}.${randomUUID()}.tmp`;
	await writeFile(temporary, bytes, { mode: 0o600 }); await rename(temporary, path);
	return bytes;
}

export async function listBlends(): Promise<SavedBlend[]> {
	let entries: string[];
	try { entries = await readdir(join(root, 'saved')); } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []; throw error; }
	const voices = await Promise.all(entries.filter(validId).map(id => readFile(join(root, 'saved', id, 'profile.json'), 'utf8').then(JSON.parse)));
	return voices.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function saveBlend(recipe: BlendRecipe, name: string): Promise<SavedBlend> {
	const bytes = await createBlend(recipe);
	const profile = { ...recipe, id: randomUUID(), name, createdAt: new Date().toISOString() };
	const folder = join(root, 'saved', `${profile.id}.tmp`);
	await mkdir(folder, { recursive: true, mode: 0o700 });
	try {
		await writeFile(join(folder, 'voice.safetensors'), bytes, { mode: 0o600 });
		await writeFile(join(folder, 'profile.json'), JSON.stringify(profile), { mode: 0o600 });
		await rename(folder, join(root, 'saved', profile.id));
	} catch (error) { await rm(folder, { recursive: true, force: true }); throw error; }
	return profile;
}

export async function savedVoice(id: string) { return readFile(join(root, 'saved', id, 'voice.safetensors')); }
export async function deleteBlend(id: string) { await rm(join(root, 'saved', id), { recursive: true, force: true }); }

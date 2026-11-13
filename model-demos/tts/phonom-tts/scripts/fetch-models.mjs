import { mkdir, readFile, rename, rm, stat } from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import { homedir } from 'node:os';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import models from '../src/lib/models.json' with { type: 'json' };

const appRoot = fileURLToPath(new URL('../', import.meta.url));
const root = resolve(process.env.PHONON_MODEL_DIR || resolve(appRoot, '.models/phonon'));
const requested = process.argv.slice(2);
const keys = requested.length ? requested : Object.keys(models).filter(key => models[key].local);
if (keys.some(key => !Object.hasOwn(models, key) || !models[key].local)) {
	console.error(`Choose one or more of: ${Object.keys(models).filter(key => models[key].local).join(', ')}`);
	process.exit(1);
}
// Credentials stay in this local setup process. No token is written into the app,
// its model files, browser requests, logs or generated bundles.
const tokenFile = process.env.HF_TOKEN_PATH || resolve(process.env.HF_HOME || resolve(homedir(), '.cache/huggingface'), 'token');
const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || (await readFile(tokenFile, 'utf8').catch(() => '')).trim();
for (const key of keys) {
	const model = models[key];
	const directory = resolve(root, key, model.revision);
	await mkdir(directory, { recursive: true });
	let failed = false;
	for (const file of ['config.json', 'tokenizer.json', 'model.q8.gguf']) {
		const target = resolve(directory, file);
		const existing = await stat(target).catch(() => undefined);
		if (existing?.isFile() && (file.endsWith('.gguf') ? existing.size === model.bytes : existing.size > 0)) continue;
		const temporary = `${target}.part`;
		try {
			const response = await fetch(`https://huggingface.co/${model.repo}/resolve/${model.revision}/${file}`, {
				headers: token ? { Authorization: `Bearer ${token}` } : {}, signal: AbortSignal.timeout(300_000)
			});
			if ([401, 403].includes(response.status)) {
				throw new Error(`Access denied. Accept access at https://huggingface.co/${model.repo} and use an authorised Hugging Face token.`);
			}
			if (!response.ok || !response.body) throw new Error(`Download failed (HTTP ${response.status}).`);
			console.log(`${key}: downloading ${file}`);
			await pipeline(Readable.fromWeb(response.body), createWriteStream(temporary));
			if (file.endsWith('.gguf')) {
				if ((await stat(temporary)).size !== model.bytes) throw new Error('Weights size differs from the pinned model.');
			} else {
				JSON.parse(await readFile(temporary, 'utf8'));
			}
			await rename(temporary, target);
		} catch (error) {
			await rm(temporary, { force: true });
			console.error(`${key}: ${error instanceof Error ? error.message : 'Download failed.'}`);
			process.exitCode = 1; failed = true; break;
		}
	}
	if (!failed) console.log(`${key}: ready`);
}

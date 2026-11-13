import { stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { env } from 'node:process';
import { models, type ModelKey } from '../models';

export const modelFiles = ['config.json', 'tokenizer.json', 'model.q8.gguf'] as const;
export function modelFile(key: ModelKey, file: string) {
	return resolve(env.PHONON_MODEL_DIR || '.models/phonon', key, models[key].revision, file);
}
export async function modelAvailability(): Promise<Record<ModelKey, boolean>> {
	return Object.fromEntries(await Promise.all(Object.entries(models).map(async ([key, model]) => {
		if (!model.local) return [key, true];
		const sizes = await Promise.all(modelFiles.map(file => stat(modelFile(key as ModelKey, file)).then(s => s.isFile() ? s.size : 0).catch(() => 0)));
		return [key, sizes[0] > 0 && sizes[1] > 0 && sizes[2] === model.bytes];
	}))) as Record<ModelKey, boolean>;
}

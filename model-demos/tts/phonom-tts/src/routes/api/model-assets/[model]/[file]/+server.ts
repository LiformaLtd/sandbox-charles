import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { error } from '@sveltejs/kit';
import { models, type ModelKey } from '../../../../../lib/models';
import { modelFile, modelFiles } from '../../../../../lib/server/model-assets';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, url }) => {
	if (!Object.hasOwn(models, params.model)) error(404, 'Unknown Phonon model.');
	const key = params.model as ModelKey;
	const model = models[key];
	if (!model.local || !modelFiles.some(file => file === params.file) || url.searchParams.get('revision') !== model.revision) {
		error(404, 'Unknown model asset or revision.');
	}
	const path = modelFile(key, params.file);
	const info = await stat(path).catch(() => undefined);
	if (!info?.isFile()) error(404, 'Download this model locally before generating speech.');
	return new Response(Readable.toWeb(createReadStream(path)) as ReadableStream<Uint8Array>, {
		headers: {
			'Content-Type': params.file.endsWith('.json') ? 'application/json' : 'application/octet-stream',
			'Content-Length': String(info.size),
			'Cache-Control': 'private, max-age=31536000, immutable',
			'X-Content-Type-Options': 'nosniff'
		}
	});
};

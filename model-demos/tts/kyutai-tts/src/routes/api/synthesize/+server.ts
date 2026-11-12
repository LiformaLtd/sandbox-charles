import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, url }) => {
	if (request.headers.get('origin') && request.headers.get('origin') !== url.origin) {
		return json({ detail: 'Use the local sandbox to run synthesis.' }, { status: 403 });
	}

	const isClone = request.headers.get('content-type')?.startsWith('multipart/form-data');
	let body: FormData | string;
	try {
		if (isClone) {
			if (Number(request.headers.get('content-length')) > 11 * 1024 * 1024) {
				return json({ detail: 'Choose a WAV or M4A file up to 10 MB.' }, { status: 413 });
			}
			const form = await request.formData();
			const reference = form.get('reference');
			if (!(reference instanceof File) || reference.size === 0 || reference.size > 10 * 1024 * 1024) {
				return json({ detail: 'Choose a WAV or M4A file up to 10 MB.' }, { status: 400 });
			}
			body = new FormData();
			body.set('reference', reference);
			body.set('text', String(form.get('text') ?? ''));
			body.set('model', String(form.get('model') ?? ''));
		} else {
			body = JSON.stringify(await request.json());
		}
	} catch {
		return json({ detail: 'Could not read the synthesis request.' }, { status: 400 });
	}

	try {
		const service = process.env.POCKET_TTS_URL || 'http://127.0.0.1:8178';
		const endpoint = new URL(isClone ? '/clone' : '/synthesize', service);
		if (url.searchParams.get('stream') === 'true') endpoint.searchParams.set('stream', 'true');
		const upstream = await fetch(endpoint, {
			method: 'POST',
			headers: isClone ? undefined : { 'content-type': 'application/json' },
			body,
			signal: AbortSignal.timeout(180_000)
		});
		const headers = new Headers({ 'cache-control': 'no-store' });
		for (const name of ['content-type', 'x-setup-ms', 'x-generation-ms', 'x-audio-seconds', 'x-model-load', 'x-sample-rate']) {
			const value = upstream.headers.get(name);
			if (value !== null) headers.set(name, value);
		}
		return new Response(upstream.body, { status: upstream.status, headers });
	} catch (error) {
		const detail = error instanceof Error && error.name === 'TimeoutError'
			? 'Inference timed out. Check the inference terminal for model download progress.'
			: 'Could not reach Pocket TTS. Start the local inference service and retry.';
		return json({ detail }, { status: 502 });
	}
};

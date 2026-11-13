import { json } from '@sveltejs/kit';
import { createBlend, listBlends, saveBlend, validateRecipe } from '../../../lib/server/voice-blends';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
	try { return json(await listBlends(), { headers: { 'Cache-Control': 'no-store' } }); }
	catch (error) { console.error('Could not list saved voice blends', error); return json({ detail: 'Could not load saved blends.' }, { status: 500 }); }
};

export const POST: RequestHandler = async ({ request, url }) => {
	if (request.headers.get('origin') !== url.origin) return json({ detail: 'Use this app to blend voices.' }, { status: 403 });
	let body, recipe;
	try {
		const raw = await request.text();
		if (raw.length > 4096) return json({ detail: 'Request too large.' }, { status: 413 });
		body = JSON.parse(raw); recipe = validateRecipe(body);
		if (body.action !== 'preview' && body.action !== 'save') throw new Error('Choose preview or save.');
		if (body.action === 'save' && (typeof body.name !== 'string' || !body.name.trim() || body.name.trim().length > 80 || /[\u0000-\u001f]/.test(body.name))) throw new Error('Enter a voice name up to 80 characters.');
	} catch (error) { return json({ detail: error instanceof Error ? error.message : 'Invalid blend request.' }, { status: 400 }); }
	try {
		if (body.action === 'save') return json(await saveBlend(recipe, body.name.trim()), { status: 201, headers: { 'Cache-Control': 'no-store' } });
		return new Response(new Uint8Array(await createBlend(recipe)), { headers: { 'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' } });
	} catch (error) { console.error('Voice blending failed', error); return json({ detail: 'Could not create the blend. Check the connection and try again.' }, { status: 502 }); }
};

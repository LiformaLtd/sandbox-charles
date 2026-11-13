import { json } from '@sveltejs/kit';
import { deleteBlend, savedVoice, validId } from '../../../../lib/server/voice-blends';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params }) => {
	if (!validId(params.id)) return json({ detail: 'Voice not found.' }, { status: 404 });
	try { return new Response(new Uint8Array(await savedVoice(params.id)), { headers: { 'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' } }); }
	catch (error) {
		if ((error as NodeJS.ErrnoException).code === 'ENOENT') return json({ detail: 'Voice not found.' }, { status: 404 });
		console.error('Could not read saved voice blend', error); return json({ detail: 'Could not read this voice.' }, { status: 500 });
	}
};
export const DELETE: RequestHandler = async ({ params, request, url }) => {
	if (request.headers.get('origin') !== url.origin) return json({ detail: 'Use this app to manage voices.' }, { status: 403 });
	if (!validId(params.id)) return json({ detail: 'Voice not found.' }, { status: 404 });
	try { await deleteBlend(params.id); return json({ deleted: true }); }
	catch (error) { console.error('Could not delete voice blend', error); return json({ detail: 'Could not delete this voice.' }, { status: 500 }); }
};

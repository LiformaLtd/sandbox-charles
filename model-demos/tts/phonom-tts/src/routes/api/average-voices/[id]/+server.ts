import { json } from '@sveltejs/kit';
import { averageVoice } from '../../../../lib/server/voice-blends';
import { averagedVoices } from '../../../../lib/voice-blends';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params }) => {
	if (!averagedVoices.some(group => group.id === params.id)) return json({ detail: 'Unknown averaged voice.' }, { status: 404 });
	try {
		return new Response(new Uint8Array(await averageVoice(params.id)), { headers: {
			'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'
		} });
	} catch (error) {
		console.error('Could not prepare averaged voice', error);
		return json({ detail: 'Could not prepare this averaged voice. Check the connection and try again.' }, { status: 502 });
	}
};

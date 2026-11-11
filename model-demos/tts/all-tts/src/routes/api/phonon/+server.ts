import { env } from 'node:process';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, url }) => {
 if (request.headers.get('origin') !== url.origin) return json({ detail: 'Use this app to generate speech.' }, { status: 403 });
 let body: { backend?: string; text?: string; voice?: string } | null;
 try {
  const raw = await request.text();
  if (raw.length > 10000) return json({ detail: 'Request too large.' }, { status: 413 });
  body = JSON.parse(raw);
 } catch { return json({ detail: 'Invalid synthesis request.' }, { status: 400 }); }
 if (!body || !['native', 'mlx'].includes(body.backend || '') || typeof body.text !== 'string' || !body.text.trim() || body.text.length > 1000 || typeof body.voice !== 'string' || !body.voice || body.voice.length > 100) {
  return json({ detail: 'Choose a runtime, voice and text up to 1,000 characters.' }, { status: 400 });
 }
 try {
  const response = await fetch(new URL('/v1/phonon/stream', env.PHONON_TTS_URL || 'http://127.0.0.1:8184'), {
   method: 'POST', headers: { 'Content-Type': 'application/json' },
   body: JSON.stringify({ backend: body.backend, text: body.text.trim(), voice: body.voice }),
   signal: AbortSignal.any([request.signal, AbortSignal.timeout(180_000)])
  });
  return new Response(response.body, { status: response.status, headers: {
   'Content-Type': response.headers.get('content-type') || 'application/x-ndjson',
   'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'
  } });
 } catch (error) {
  if (request.signal.aborted) return new Response(null, { status: 499 });
  console.error('Phonon local service request failed', error);
  return json({ detail: 'Could not reach Phonon. Start the Phonon service on port 8184 and try again.' }, { status: 502 });
 }
};

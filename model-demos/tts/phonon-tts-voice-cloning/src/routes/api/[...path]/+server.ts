import { env } from 'node:process';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const uuid = '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}';
const readable = new RegExp(`^(health|voices|voices/${uuid}/(reference|embedding|original-comparison|fitted-comparison)|audio/${uuid})$`);
const deletable = new RegExp(`^voices/${uuid}$`);
const writable = new RegExp(`^(voices|speech|voices/${uuid}/(transcribe|refine|compare)|(refinements|evaluations)/${uuid}/cancel)$`);

const proxy: RequestHandler = async ({ request, params, url }) => {
  const path = params.path || '';
  const method = request.method;
  if (!(method === 'GET' && readable.test(path)) && !(method === 'POST' && writable.test(path)) && !(method === 'DELETE' && deletable.test(path))) {
    return json({ detail: 'Not found.' }, { status: 404 });
  }
  if (method !== 'GET' && request.headers.get('origin') !== url.origin) {
    return json({ detail: 'Use this app to create and manage voices.' }, { status: 403 });
  }
  const headers = new Headers({ 'x-lab-token': env.PHONON_CLONE_TOKEN || '' });
  for (const name of ['content-type', 'x-voice-consent', 'range']) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  let body: Uint8Array | undefined;
  if (method === 'POST') {
    const reader = request.body?.getReader();
    if (!reader && !path.endsWith('/transcribe') && !path.endsWith('/cancel')) return json({ detail: 'Request body missing.' }, { status: 400 });
    const chunks: Uint8Array[] = [];
    let length = 0;
    const limit = path === 'voices' ? 2_000_000 : 10_000;
    while (reader) {
      const next = await reader.read();
      if (next.done) break;
      length += next.value.length;
      if (length > limit) {
        await reader.cancel();
        return json({ detail: 'Request too large.' }, { status: 413 });
      }
      chunks.push(next.value);
    }
    body = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) { body.set(chunk, offset); offset += chunk.length; }
  }
  const target = new URL(`/${path}`, env.PHONON_CLONE_URL || 'http://127.0.0.1:8186');
  if (path === 'voices' && method === 'POST') target.searchParams.set('name', url.searchParams.get('name') || 'My voice');
  try {
    const response = await fetch(target, { method, headers, body: body as BodyInit | undefined,
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(path.endsWith('/refine') || path.endsWith('/compare') ? 900_000 : 180_000)]) });
    const outgoing = new Headers({ 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    for (const name of ['content-type', 'content-length', 'content-range', 'accept-ranges', 'content-disposition']) {
      const value = response.headers.get(name);
      if (value) outgoing.set(name, value);
    }
    return new Response(response.body, { status: response.status, headers: outgoing });
  } catch (error) {
    console.error('Local voice-cloning service request failed', error);
    return json({ detail: 'The local voice engine is unavailable. Start this app with npm run dev.' }, { status: 502 });
  }
};

export const GET = proxy;
export const POST = proxy;
export const DELETE = proxy;

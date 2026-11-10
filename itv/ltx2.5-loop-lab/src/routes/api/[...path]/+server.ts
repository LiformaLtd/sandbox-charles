import { LTX25_API_URL } from '$app/env/private';
import type { RequestHandler } from './$types';

const proxy: RequestHandler = async ({ request, params, url }) => {
  const origin = request.headers.get('origin');
  if (request.method === 'POST' && origin && origin !== url.origin) {
    return Response.json({ detail: 'Submit jobs from this Loop Lab page.' }, { status: 403 });
  }
  const path = params.path ?? '';
  if (!/^(health|v1\/loop-sets(?:\/[a-f0-9]{64}(?:\/(image|retry|archive|regenerate-idle))?)?|v1\/jobs\/[a-f0-9-]{36}\/(video|log))$/.test(path)) {
    return Response.json({ detail: 'Unknown local model endpoint' }, { status: 404 });
  }
  const headers = new Headers();
  for (const key of ['content-type', 'range']) {
    const value = request.headers.get(key);
    if (value) headers.set(key, value);
  }
  try {
    let body: ArrayBuffer | undefined;
    if (request.method === 'POST') {
      body = await request.arrayBuffer();
      if (body.byteLength > 22 * 1024 * 1024) return Response.json({ detail: 'Upload is too large' }, { status: 413 });
    }
    // SvelteKit's event.fetch inherits the browser Origin even with fresh
    // headers. This is a server-to-server request, so use native fetch with
    // only the explicit headers above. Browser same-origin checks stay here.
    const response = await globalThis.fetch(`${LTX25_API_URL}/${path}`, {
      method: request.method, headers, body
    });
    const resultHeaders = new Headers();
    for (const key of ['content-type', 'content-length', 'content-range', 'accept-ranges', 'content-disposition']) {
      const value = response.headers.get(key);
      if (value) resultHeaders.set(key, value);
    }
    resultHeaders.set('cache-control', 'no-store');
    return new Response(response.body, { status: response.status, headers: resultHeaders });
  } catch (error) {
    console.error('Loop Lab could not contact the local model API', error);
    return Response.json({ detail: 'Local model server is offline. Start the LTX-2.5 API on port 3026.' }, { status: 503 });
  }
};
export const GET = proxy;
export const POST = proxy;

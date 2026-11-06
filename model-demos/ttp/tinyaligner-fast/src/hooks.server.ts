import type { Handle } from '@sveltejs/kit/hooks';

export const handle: Handle = async ({ event, resolve }) => {
  const response = await resolve(event);
  if (event.url.pathname === '/benchmark' && process.env.BENCHMARK_ISOLATION === '1') {
    response.headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    response.headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  }
  return response;
};

import { defineEnvVars } from '@sveltejs/kit/env';
export const variables = defineEnvVars({
  LTX25_API_URL: { schema: (value: string | undefined) => {
    const address = value || 'http://127.0.0.1:3026';
    const url = new URL(address);
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('LTX25_API_URL must be an HTTP URL');
    return address.replace(/\/$/, '');
  } }
});

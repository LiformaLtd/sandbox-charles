import { createServer } from 'vite';

export default async function startServer() {
  if (process.env.TEST_BASE_URL) return;
  const server = await createServer({ server: { host: '127.0.0.1', port: 5173, strictPort: false } });
  await server.listen();
  process.env.LTX_TEST_URL = server.resolvedUrls!.local[0];
  return async () => { await server.close(); };
}

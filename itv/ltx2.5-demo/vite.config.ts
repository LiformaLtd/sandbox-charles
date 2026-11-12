import { defineConfig, loadEnv } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
    plugins: [svelte()],
    server: { proxy: { '/api': {
      target: env.LTX25_API_URL || 'http://127.0.0.1:3026',
      rewrite: (path) => path.replace(/^\/api/, '')
    } } }
  };
});

import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit({ adapter: adapter(), paths: { origin: 'http://127.0.0.1:5180' } })],
	worker: { format: 'es' },
	server: { host: '127.0.0.1', port: 5180, strictPort: true },
	preview: { host: '127.0.0.1', port: 5180, strictPort: true }
});

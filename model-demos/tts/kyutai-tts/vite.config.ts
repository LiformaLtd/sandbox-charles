import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	worker: { format: 'es' },
	plugins: [sveltekit({ adapter: adapter() })],
	server: { host: '127.0.0.1', port: 5178, strictPort: true },
	preview: { host: '127.0.0.1', port: 5178, strictPort: true }
});

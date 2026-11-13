import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit({ adapter: adapter() })],
	worker: { format: 'es' },
	optimizeDeps: { include: ['phonon-tts', 'phonon-tts/worker.js'] },
	server: { host: '127.0.0.1', port: 5173, strictPort: false },
	preview: { host: '127.0.0.1', port: 5173, strictPort: false }
});

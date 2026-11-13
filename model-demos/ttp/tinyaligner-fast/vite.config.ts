import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const benchmarkHeaders = process.env.BENCHMARK_ISOLATION === '1'
	? { 'Cross-Origin-Embedder-Policy': 'require-corp' }
	: undefined;

export default defineConfig({
	plugins: [sveltekit({ adapter: adapter() })],
	worker: { format: 'es' },
	server: { host: '127.0.0.1', port: 5173, strictPort: false, headers: benchmarkHeaders },
	preview: { host: '127.0.0.1', port: 5173, strictPort: false, headers: benchmarkHeaders }
});

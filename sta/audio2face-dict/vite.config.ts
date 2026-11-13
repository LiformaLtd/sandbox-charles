import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit({ adapter: adapter() })],
	worker: { format: 'es' },
	build: {
		rolldownOptions: {
			output: { codeSplitting: { groups: [{ name: 'three-core', test: /three[\/]build[\/]three\.core\.js$/ }] } }
		}
	},
	optimizeDeps: { include: ['phonemize'], exclude: ['onnxruntime-web/wasm', 'phonon-tts/worker.js'] },
	server: { host: '127.0.0.1', port: 5173, strictPort: false },
	preview: { host: '127.0.0.1', port: 5173, strictPort: false }
});

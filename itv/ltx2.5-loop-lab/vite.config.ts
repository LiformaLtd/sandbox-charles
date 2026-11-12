import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
const testing = process.env.LOOPLAB_TEST_RUN === '1';
export default defineConfig({
  cacheDir: testing ? 'node_modules/.vite-loop-tests' : 'node_modules/.vite',
  plugins:[sveltekit({adapter:adapter(),outDir:testing ? '.svelte-kit-tests' : '.svelte-kit'})]
});

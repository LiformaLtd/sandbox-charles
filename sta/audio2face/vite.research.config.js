import { fileURLToPath } from 'node:url';
import { defineConfig, mergeConfig } from 'vite';
import baseline from './vite.config.js';

export default defineConfig((environment) => mergeConfig(baseline(environment), {
  plugins: [{
    name: 'expressive-research-entry',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url === '/' || request.url?.startsWith('/?')) {
          response.writeHead(302, { Location: '/expressive.html' });
          response.end();
          return;
        }
        next();
      });
    },
  }],
  build: {
    outDir: 'dist-expressive',
    rollupOptions: { input: fileURLToPath(new URL('./expressive.html', import.meta.url)) },
  },
}));

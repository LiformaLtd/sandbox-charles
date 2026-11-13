import { defineConfig } from 'vite';

export default defineConfig({
  server: { host: '127.0.0.1', port: 5173, strictPort: false },
  preview: { host: '127.0.0.1', port: 5173, strictPort: false },
  build: {
    rollupOptions: {
      // Load Spark's prebuilt renderer on demand and share its Three.js instance.
      external: (id) => id === '@sparkjsdev/spark' || id === 'three' || id.startsWith('three/addons/'),
    },
  },
});

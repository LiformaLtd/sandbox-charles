import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      // Load Spark's prebuilt renderer on demand and share its Three.js instance.
      external: (id) => id === '@sparkjsdev/spark' || id === 'three' || id.startsWith('three/addons/'),
    },
  },
});

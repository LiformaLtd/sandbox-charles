import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', workers: 1, use: { channel: 'chrome' },
  globalSetup: './tests/start-server.ts'
});

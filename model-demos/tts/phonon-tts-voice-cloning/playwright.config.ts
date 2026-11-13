import { defineConfig } from '@playwright/test';
import { resolve } from 'node:path';

export default defineConfig({
  testDir: './tests/browser', timeout: 90_000, workers: 1, reporter: 'list',
  use: {
    baseURL: process.env.TEST_BASE_URL || 'http://127.0.0.1:5173',
    viewport: { width: 1360, height: 1100 }, permissions: ['microphone'],
    launchOptions: {
      executablePath: process.env.CHROME_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
        `--use-file-for-fake-audio-capture=${resolve(process.env.TEST_VOICE_WAV || '.data/testing/microphone.wav')}`,
        '--autoplay-policy=no-user-gesture-required']
    }
  }
});

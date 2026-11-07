// Native-ORT reference tracks must cover the complete streaming crop, not one window.
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import assert from 'node:assert/strict';
const { webkit, devices } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const base = process.env.DIAGNOSTIC_URL;
if (!base) throw Error('Set DIAGNOSTIC_URL to a hosted-mode Vite source server.');
const root = resolve(process.env.A2F_ACTIVATION_OUTPUT || '.local/packed-diagnostic');
const assetPath = process.env.A2F_ACTIVATION_ASSET_PATH || '/.local/packed-diagnostic/';
const cases = JSON.parse(await readFile(join(root, 'parity-cases.json')));
const modelUrls = process.env.A2F_PARITY_MODELS ? JSON.parse(process.env.A2F_PARITY_MODELS)
  : Object.fromEntries(['v3', 'v4'].map(name => [name, '/assets/james-arkit-student-' + name + '.onnx']));
const browser = await webkit.launch(process.env.WEBKIT_EXECUTABLE ? { executablePath: process.env.WEBKIT_EXECUTABLE } : {});
const report = [];
try {
  for (const [model, modelUrl] of Object.entries(modelUrls)) {
    const page = await browser.newPage({ ...devices['iPhone 13'] });
    await page.goto(base + '/?model=compact-' + model);
    const result = await page.evaluate(async ({ cases, model, modelUrl, assetPath, before }) => {
      const { createCompactWebGpu } = await import('/src/compact-webgpu.js');
      const { modelAssetUrl } = await import('/src/model-assets.js');
      const { audioWindows } = await import('/src/geometry.mjs');
      const adapter = await navigator.gpu.requestAdapter();
      const bytes = await (await fetch(modelAssetUrl(modelUrl))).arrayBuffer();
      const runners = { fixed: await createCompactWebGpu(adapter, bytes, () => {}) };
      if (before) {
        const original = await import(assetPath + 'compact-original.js');
        runners.original = await original.createCompactWebGpu(await navigator.gpu.requestAdapter(), bytes, () => {});
      }
      const output = [];
      for (const item of cases) {
        const pcm = new Float32Array(await (await fetch(assetPath + item.audio)).arrayBuffer());
        const expected = new Float32Array(await (await fetch(assetPath + item.reference[model])).arrayBuffer());
        for (const [variant, run] of Object.entries(runners)) {
          const track = new Float32Array(expected.length), timings = [];
          let cursor = 0;
          async function* chunks() { for (let i = 0; i < pcm.length; i += 1280) yield pcm.slice(i, i + 1280); }
          for await (const window of audioWindows(chunks())) {
            if (!window.sampleIndices.length) continue;
            const start = performance.now(), values = await run(window.samples);
            timings.push(performance.now() - start);
            const crop = values.subarray(window.first * 52, (window.first + window.sampleIndices.length) * 52);
            track.set(crop, cursor); cursor += crop.length;
          }
          if (cursor !== expected.length) throw Error('Incomplete streaming output');
          let max = 0, sum = 0, nonfinite = 0;
          const zeroedFrames = [], badFrames = [];
          for (let frame = 0; frame < track.length / 52; frame++) {
            let largest = 0, actualMax = 0, expectedMax = 0;
            for (let c = 0; c < 52; c++) {
              const i = frame * 52 + c, error = Math.abs(track[i] - expected[i]);
              if (!Number.isFinite(track[i])) nonfinite++;
              max = Math.max(max, error); sum += error; largest = Math.max(largest, error);
              actualMax = Math.max(actualMax, Math.abs(track[i])); expectedMax = Math.max(expectedMax, Math.abs(expected[i]));
            }
            if (largest > .00005) badFrames.push(frame);
            if (actualMax === 0 && expectedMax > .0001) zeroedFrames.push(frame);
          }
          const raw = new Uint8Array(track.buffer), parts = [];
          for (let i = 0; i < raw.length; i += 16384) parts.push(String.fromCharCode(...raw.subarray(i, i + 16384)));
          timings.shift(); timings.sort((a, b) => a - b);
          output.push({ name: item.name, model, variant, values: track.length, max, mae: sum / track.length,
            nonfinite, zeroedFrames, badFrames, median_ms: timings[Math.floor(timings.length / 2)],
            base64: btoa(parts.join('')) });
        }
      }
      return output;
    }, { cases, model, modelUrl, assetPath, before: process.env.COMPARE_ORIGINAL === '1' });
    for (const row of result) {
      await writeFile(join(root, row.name + '-' + row.variant + '-' + model + '.f32'), Buffer.from(row.base64, 'base64'));
      delete row.base64; report.push(row);
      console.log(row.name, model, row.variant, 'max error', row.max, 'zeroed frames', row.zeroedFrames.length);
      if (row.variant === 'fixed') {
        assert.equal(row.nonfinite, 0); assert.equal(row.zeroedFrames.length, 0);
        assert.ok(row.max < .00005, 'Native/GPU mismatch');
      }
    }
    await page.close();
  }
  await writeFile(join(root, 'activation-parity.json'), JSON.stringify(report, null, 2));
} finally { await browser.close(); }

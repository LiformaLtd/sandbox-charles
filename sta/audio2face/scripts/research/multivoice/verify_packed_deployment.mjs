// Inspect actual compact GPU readbacks during the reported plosive regression.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
const { webkit, devices } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const base = process.env.DEPLOYMENT_URL;
if (!base) throw Error('Set DEPLOYMENT_URL.');
const text = await readFile('.local/packed-diagnostic/text.txt', 'utf8');
const results = [];
const browser = await webkit.launch(process.env.WEBKIT_EXECUTABLE ? { executablePath: process.env.WEBKIT_EXECUTABLE } : {});
try {
  for (const engine of ['phonon', 'supertonic']) for (const model of ['compact-v3', 'compact-v4']) {
    const page = await browser.newPage({ ...devices['iPhone 13'] });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.addInitScript(() => {
      window.readbacks = [];
      const original = GPUBuffer.prototype.getMappedRange;
      GPUBuffer.prototype.getMappedRange = function (...args) {
        const buffer = original.apply(this, args);
        if (this.size === 60 * 52 * 4) {
          const values = new Float32Array(buffer);
          const zeroFrames = [];
          for (let frame = 0; frame < 60; frame++) {
            if (values.subarray(frame * 52, (frame + 1) * 52).every(x => x === 0)) zeroFrames.push(frame);
          }
          window.readbacks.push({ zeroFrames, finite: values.every(Number.isFinite) });
        }
        return buffer;
      };
    });
    try {
      await page.goto(`${base}/?model=${model}&sta=webgpu`);
      await page.waitForFunction(() => !document.querySelector('#generate').disabled);
      assert.equal(await page.locator('input[name=runtime]:checked').inputValue(), 'webgpu');
      await page.locator(`input[name=engine][value=${engine}]`).check();
      await page.locator('#speech').fill(text);
      await page.locator('#generate').click();
      await page.waitForFunction(() => document.querySelector('#status').textContent === 'Animation ready to replay.'
        || document.querySelector('#status').classList.contains('error'), null, { timeout: 150000 });
      assert.equal(await page.locator('#status').textContent(), 'Animation ready to replay.');
      const readbacks = await page.evaluate(() => window.readbacks);
      assert.ok(readbacks.length > 20, 'Expected full paragraph GPU inference');
      // Compact skips the empty warm-up. The first emitted crop is 30:45;
      // subsequent windows use 15:45 (the last may contain trailing padding).
      const bad = readbacks.flatMap((r, index) => r.zeroFrames
        .filter(frame => frame >= (index === 0 ? 30 : 15) && frame < 45).map(frame => ({ index, frame })));
      assert.deepEqual(bad, []);
      assert.ok(readbacks.every(r => r.finite));
      assert.deepEqual(errors, []);
      const result = { engine, model, windows: readbacks.length, badFrames: bad, errors };
      results.push(result); console.log(JSON.stringify(result));
    } finally { await page.close(); }
  }
  await writeFile('.local/packed-diagnostic/hosted-packed.json', JSON.stringify({ base, results }, null, 2));
} finally { await browser.close(); }

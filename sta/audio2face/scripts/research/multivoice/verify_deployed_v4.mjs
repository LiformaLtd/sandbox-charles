// Run against the actual hosted build, with iPhone WebKit defaults and CDN assets.
import { writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import assert from 'node:assert/strict';
const { webkit, devices } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const base = process.env.DEPLOYMENT_URL;
if (!base) throw Error('Set DEPLOYMENT_URL.');
const output = resolve(process.env.A2F_VALIDATION_OUTPUT || '.local/compact-v4-validation');
const browser = await webkit.launch(process.env.WEBKIT_EXECUTABLE ? { executablePath: process.env.WEBKIT_EXECUTABLE } : {});
try {
  const page = await browser.newPage({ ...devices['iPhone 13'], userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1' });
  const errors = [], requests = [], runs = [];
  page.on('pageerror', e => errors.push(e.message)); page.on('request', r => requests.push(r.url()));
  await page.addInitScript(() => {
    let poses = 0, pipelines = 0, audio = 0;
    const draw = WebGL2RenderingContext.prototype.bufferSubData;
    WebGL2RenderingContext.prototype.bufferSubData = function (...a) { poses++; return draw.apply(this, a); };
    for (const k of ['createComputePipeline', 'createComputePipelineAsync']) {
      const f = GPUDevice.prototype[k]; GPUDevice.prototype[k] = function (...a) { pipelines++; return f.apply(this, a); };
    }
    const start = AudioBufferSourceNode.prototype.start;
    AudioBufferSourceNode.prototype.start = function (...a) { audio++; return start.apply(this, a); };
    window.qa = () => ({ poses, pipelines, audio });
  });
  const ready = () => page.waitForFunction(() => !document.querySelector('#generate').disabled, null, { timeout: 60000 });
  await page.goto(base + '/?model=compact-v4'); await ready();
  assert.equal(await page.locator('input[name=runtime]:checked').inputValue(), 'webgpu');
  const text = 'Please bring my blue bag before we leave. We should check the map together.';
  await page.locator('#speech').fill(text);
  for (const model of ['compact-v4', 'compact-v3', 'compact-v4']) {
    if (await page.locator('input[name=animation]:checked').inputValue() !== model) {
      await page.locator('input[name=animation][value=' + model + ']').check();
      await page.waitForURL(u => u.searchParams.get('model') === model); await ready();
    }
    assert.equal(await page.locator('#speech').inputValue(), text);
    assert.equal(await page.locator('input[name=runtime]:checked').inputValue(), 'webgpu');
    const at = requests.length;
    await page.locator('#generate').click();
    await page.waitForFunction(() => document.querySelector('#status').textContent === 'Animation ready to replay.'
      || document.querySelector('#status').classList.contains('error'), null, { timeout: 90000 });
    await ready(); assert.equal(await page.locator('#status').textContent(), 'Animation ready to replay.');
    const stats = await page.evaluate(() => qa()); assert.equal(stats.pipelines, 2); assert.ok(stats.audio > 0);
    await page.waitForFunction(n => qa().poses > n + 10, stats.poses);
    const filename = model === 'compact-v4' ? 'james-arkit-student-v4.onnx' : 'james-arkit-student-v3.onnx';
    assert.ok(requests.slice(at).some(u => u.includes(filename)), 'Wrong model asset');
    runs.push({ model, stats, duration: await page.locator('#duration').textContent(), idle: true });
  }
  if (process.env.TEST_SUPERTONIC === '1') {
    await page.locator('input[name=engine][value=supertonic]').check();
    await page.locator('#speech').fill('Please keep a spare paper copy of the complete proposal.');
    await page.locator('#generate').click();
    await page.waitForFunction(() => document.querySelector('#status').textContent === 'Animation ready to replay.'
      || document.querySelector('#status').classList.contains('error'), null, { timeout: 120000 });
    await ready(); assert.equal(await page.locator('#status').textContent(), 'Animation ready to replay.');
    runs.push({ model: 'compact-v4', engine: 'supertonic', duration: await page.locator('#duration').textContent() });
  }
  assert.deepEqual(errors, []);
  await page.screenshot({ path: join(output, 'deployed-v4-iphone.png'), fullPage: true });
  const result = { base, runs, errors, modelUrls: requests.filter(u => /james-arkit-student-v[34]\.onnx/.test(u)) };
  await writeFile(join(output, 'deployed-v4-smoke.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} finally { await browser.close(); }

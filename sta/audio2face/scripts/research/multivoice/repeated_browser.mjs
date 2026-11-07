// Full-page WebGPU checks: three distinct generations without tearing down models.
import { execFileSync, spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import assert from 'node:assert/strict';
const { webkit, devices } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const root = resolve(process.env.A2F_VALIDATION_OUTPUT || '.local/compact-v4-validation', 'browser');
const base = process.env.BENCH_URL;
if (!base) throw Error('Set BENCH_URL to the isolated validation harness.');
const texts = [
  'Meet me in the morning. We can make a plan together.',
  'Meet me in the morning. We can make a plan together. After breakfast, we will pack the blue bags, check the map, and walk down to the river. The path goes past the old bridge, where we can stop and watch the water before heading home.',
  'Please bring my blue bag before we leave. We should check the map together.',
];
const processInfo = () => execFileSync('/bin/ps', ['-axo', 'pid=,comm='], { encoding: 'utf8' })
  .split('\n').map(line => line.trim().match(/^(\d+)\s+(.*)$/)).filter(Boolean)
  .map(([, id, command]) => ({ id: Number(id), command }));
const launch = () => webkit.launch(process.env.WEBKIT_EXECUTABLE ? { executablePath: process.env.WEBKIT_EXECUTABLE } : {});
const ua = 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1';

// Parity gets its own process and cannot warm any measured model session.
{
  const browser = await launch();
  try {
    const page = await browser.newPage({ ...devices['iPhone 13'], userAgent: ua });
    await page.goto(base + '/parity.html?candidate=candidate');
    await page.waitForFunction(() => ['Passed', 'Failed'].includes(document.querySelector('#status').textContent), null, { timeout: 60000 });
    const result = await page.locator('#result').textContent();
    assert.equal(await page.locator('#status').textContent(), 'Passed', result);
    await writeFile(join(root, 'candidate-browser-parity.json'), result);
  } finally { await browser.close(); }
}

for (const engine of ['phonon', 'supertonic']) for (const candidate of ['baseline', 'candidate']) {
  const before = new Set(processInfo().map(p => p.id));
  const browser = await launch();
  let sampler;
  try {
    const page = await browser.newPage({ ...devices['iPhone 13'], userAgent: ua });
    const errors = [], consoleMessages = [], samples = [], stages = [], plays = [];
    let navigations = 0;
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (['error', 'warning'].includes(message.type())) consoleMessages.push({ type: message.type(), text: message.text() }); });
    page.on('framenavigated', frame => { if (frame === page.mainFrame()) navigations++; });
    await page.addInitScript(() => {
      let poses = 0, buffers = 0, pipelines = 0, gpuLive = 0, gpuPeak = 0, audioStarts = 0, audioEnergy = 0;
      const sizes = new WeakMap();
      const update = WebGL2RenderingContext.prototype.bufferSubData;
      WebGL2RenderingContext.prototype.bufferSubData = function (...args) { poses++; return update.apply(this, args); };
      const create = GPUDevice.prototype.createBuffer, destroy = GPUBuffer.prototype.destroy;
      GPUDevice.prototype.createBuffer = function (...args) {
        const value = create.apply(this, args); sizes.set(value, value.size);
        gpuLive += value.size; gpuPeak = Math.max(gpuPeak, gpuLive); buffers++; return value;
      };
      GPUBuffer.prototype.destroy = function () { gpuLive -= sizes.get(this) ?? 0; sizes.delete(this); return destroy.call(this); };
      for (const key of ['createComputePipeline', 'createComputePipelineAsync']) {
        const original = GPUDevice.prototype[key];
        GPUDevice.prototype[key] = function (...args) { pipelines++; return original.apply(this, args); };
      }
      const start = AudioBufferSourceNode.prototype.start;
      AudioBufferSourceNode.prototype.start = function (...args) {
        audioStarts++;
        const pcm = this.buffer?.getChannelData(0);
        if (pcm) for (let i = 0; i < pcm.length; i += 128) audioEnergy += pcm[i] * pcm[i];
        return start.apply(this, args);
      };
      window.staMs = [];
      window.stats = () => ({ poses, buffers, pipelines, gpuLive, gpuPeak, audioStarts, audioEnergy, staMs: [...window.staMs] });
    });
    await page.goto(base + '/?model=compact-v3&sta=webgpu&candidate=' + candidate);
    const ready = () => page.waitForFunction(() => !document.querySelector('#generate').disabled, null, { timeout: 60000 });
    await ready();
    const pids = processInfo().filter(p => !before.has(p.id) && /WebKit.*(WebContent|GPU)/.test(p.command));
    assert.equal(pids.length, 2, JSON.stringify(pids));
    sampler = spawn(join(root, 'sample'), pids.map(p => String(p.id)));
    sampler.on('error', error => errors.push('Sampler: ' + error.message));
    createInterface({ input: sampler.stdout }).on('line', line => {
      const [time, ...values] = line.split(' ');
      samples.push({ at: Number(time), processes: values.map(v => v.split(':').map(Number)) });
    });
    const mark = async name => {
      const state = await page.evaluate(() => ({ at: Date.now(), status: document.querySelector('#status').textContent,
        duration: document.querySelector('#duration').textContent, ...stats() }));
      stages.push({ name, ...state }); return stages.at(-1);
    };
    await mark('idle');
    await page.locator('input[name=engine][value=' + engine + ']').check();
    await page.locator('#prepare-tts').click();
    await page.waitForFunction(() => /is ready\.$/.test(document.querySelector('#status').textContent)
      || document.querySelector('#status').classList.contains('error'), null, { timeout: 60000 });
    assert.equal(await page.locator('#status').textContent(), engine === 'phonon' ? 'Phonon is ready.' : 'Supertonic 3 is ready.');
    await ready(); await mark('tts');
    await page.locator('#prepare-sta').click();
    await page.waitForFunction(() => document.querySelector('#status').textContent === 'STA is ready.'
      || document.querySelector('#status').classList.contains('error'), null, { timeout: 60000 });
    assert.equal(await page.locator('#status').textContent(), 'STA is ready.');
    await ready(); const prepared = await mark('both');
    assert.equal(prepared.buffers, 25); assert.equal(prepared.pipelines, 2);
    await page.evaluate(() => {
      window.lastRenderedFrame = -1;
      new MutationObserver(() => {
        const value = Number(document.querySelector('#viewport').dataset.frame);
        if (Number.isFinite(value)) lastRenderedFrame = Math.max(lastRenderedFrame, value);
      }).observe(document.querySelector('#viewport'), { attributes: true, attributeFilter: ['data-frame'] });
    });
    for (const [index, text] of texts.entries()) {
      await page.locator('#speech').fill(text);
      await page.evaluate(() => { lastRenderedFrame = -1; staMs = []; });
      const begin = await mark('play-' + (index + 1) + '-start');
      await page.locator('#generate').click();
      await page.waitForFunction(() => document.querySelector('#status').textContent === 'Animation ready to replay.'
        || document.querySelector('#status').classList.contains('error'), null, { timeout: 90000 });
      await ready(); const end = await mark('play-' + (index + 1) + '-end');
      assert.equal(end.status, 'Animation ready to replay.');
      assert.ok(end.audioStarts > begin.audioStarts && end.audioEnergy > begin.audioEnergy, 'Missing fresh non-silent audio');
      const frame = await page.evaluate(() => lastRenderedFrame); assert.ok(frame > 60);
      assert.ok(end.staMs.length > 2, 'Missing fresh STA inference');
      assert.equal(end.buffers, prepared.buffers); assert.equal(end.pipelines, prepared.pipelines);
      assert.equal(end.gpuLive, prepared.gpuLive);
      await page.waitForFunction(n => stats().poses > n + 10, end.poses);
      const idle = await mark('play-' + (index + 1) + '-idle');
      const sampled = samples.filter(s => s.at >= begin.at && s.at <= idle.at);
      assert.ok(sampled.length > 10, 'Insufficient process samples');
      const peakMB = Math.max(...sampled.map(s => s.processes.reduce((n, p) => n + Math.max(0, p[1]), 0))) / 1e6;
      const times = end.staMs.slice(1).sort((a, b) => a - b);
      plays.push({ index: index + 1, text, duration: end.duration, lastFrame: frame,
        warm_block_ms: times[Math.floor(times.length / 2)], peak_MB: peakMB,
        gpu_live_bytes: end.gpuLive, buffers: end.buffers, pipelines: end.pipelines, idle_after: idle.poses > end.poses });
    }
    assert.equal(navigations, 1); assert.deepEqual(errors, []);
    const summary = { candidate, engine, plays, navigations,
      whole_sequence_peak_MB: Math.max(...samples.map(s => s.processes.reduce((n, p) => n + Math.max(0, p[1]), 0))) / 1e6,
      selected_model: JSON.parse(await readFile(join(root, 'selected-model.json'))) };
    await writeFile(join(root, 'repeated-' + candidate + '-' + engine + '.json'), JSON.stringify({ summary, stages, samples, consoleMessages }, null, 2));
    console.log(JSON.stringify(summary), '\n');
  } finally { sampler?.kill(); await browser.close(); }
}

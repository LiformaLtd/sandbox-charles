import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { setTimeout } from 'node:timers/promises';
import { chromium } from '@playwright/test';
import { SentencePieceProcessor } from '@sctg/sentencepiece-js';

// The real Pocket model's tokenizer, not speech weights or private recordings.
const revision = '9b874ef8e0be4a95a352a0fe0455e064e75404e0';
const response = await fetch(`https://huggingface.co/maxime/pocket-tts-18m-onnx/resolve/${revision}/english_18m_refl_2026-10/tokenizer.model`, { signal: AbortSignal.timeout(60_000) });
assert.ok(response.ok, `Tokenizer download failed: HTTP ${response.status}`);
const model = Buffer.from(await response.arrayBuffer());
assert.equal(createHash('sha256').update(model).digest('hex'), 'd461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6');
const original = new SentencePieceProcessor();
await original.loadFromB64StringModel(model.toString('base64'));
const phrases = [
  'Hello from Pocket TTS.', 'I have an appointment at three o’clock.',
  'Emma moved my camera from the middle shelf.', 'The price is $12.50.',
  'Call 01234 567890.', 'Dr. Smith arrives at 3:45 p.m.',
  "Don't forget your keys!", 'Really? Yes… absolutely.',
  'One; two, three: four.', 'Café, naïve, résumé.', '你好。', '👋 Hello!',
  '', ' ', '\n\t', '  Multiple   spaces.  ', 'Line one.\nLine two.',
  'A', 'ABC', '123', '0.001', 'a-b–c—d', '“Quoted” and "quoted".',
  'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz',
  'The quick brown fox jumps over the lazy dog.', 'She sells seashells by the seashore.',
  'My email is test@example.com.', 'Visit https://example.com/path?a=1&b=2.',
  'Angry, happy, sad, excited.', 'First\u00a0second.', 'Version 2.3.1.'
];
const expected = phrases.map((text) => original.encodeIds(text));
const root = new URL('../', import.meta.url);
const { FORCE_COLOR, NO_COLOR, ...env } = process.env;
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5173'], { cwd: root, env, stdio: ['ignore', 'pipe', 'pipe'] });
let logs = '';
for (const stream of [server.stdout, server.stderr]) stream.on('data', (data) => { logs = (logs + data).slice(-20_000); });
let browser;
try {
  let ready = false;
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(`Test server exited: ${logs}`);
    const address = logs.replace(/\u001b\[[0-9;]*m/g, '').match(/Local:\s+(http:\/\/127\.0\.0\.1:\d+\/)/)?.[1];
    try { if (address && (await fetch(address, { signal: AbortSignal.timeout(500) })).ok) { ready = true; break; } } catch {}
    await setTimeout(200);
  }
  assert.ok(ready, `Test server did not become ready: ${logs}`);
  browser = await chromium.launch({ executablePath: process.env.CHROME_EXECUTABLE ?? (process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : undefined), headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await page.route('**/__tokenizer_fixture', (route) => route.fulfill({ body: model, contentType: 'application/octet-stream' }));
  const address = logs.replace(/\u001b\[[0-9;]*m/g, '').match(/Local:\s+(http:\/\/127\.0\.0\.1:\d+\/)/)?.[1];
  assert.ok(address, 'Vite did not print its actual URL.');
  await page.goto(address);
  assert.match(await page.title(), /TTS/i);
  const actual = await page.evaluate(async (inputs) => {
    const { createTokenizer } = await import('/src/lib/pocket/onnx/tokenizer.ts');
    const tokenizer = await createTokenizer(await (await fetch('/__tokenizer_fixture')).arrayBuffer());
    return inputs.map((text) => tokenizer.encodeIds(text));
  }, phrases);
  assert.deepEqual(actual, expected);
  assert.deepEqual(errors, []);
  console.log(`Chrome: lab renders without console errors; browser token IDs match upstream for ${phrases.length} cases.`);
} finally {
  await browser?.close();
  server.kill('SIGTERM');
}

import { writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import assert from 'node:assert/strict';
const { webkit } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const root = resolve(process.env.A2F_VALIDATION_OUTPUT || '.local/compact-v4-validation');
if (!process.env.BENCH_URL) throw Error('Set BENCH_URL to the validation harness.');
const browser = await webkit.launch(process.env.WEBKIT_EXECUTABLE ? { executablePath: process.env.WEBKIT_EXECUTABLE } : {});
try {
  const page = await browser.newPage({ viewport: { width: 1800, height: 1050 } });
  const errors = [], cases = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.env.BENCH_URL + '/review.html');
  const ready = () => page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('Ready.'), null, { timeout: 30000 });
  await ready();
  assert.equal(await page.locator('canvas').count(), 3);
  assert.equal(await page.locator('#clip option').count(), 6);
  for (let i = 0; i < 6; i++) {
    await page.locator('#clip').selectOption(String(i)); await ready();
    await page.locator('#play').click();
    await page.waitForFunction(() => document.querySelector('#audio').currentTime > .15);
    await page.locator('#play').click();
    const button = page.locator('#snapshots button').nth(1);
    const expected = Number(await button.getAttribute('data-frame'));
    await button.click();
    // Seeking is asynchronous: a button click alone does not prove the frame
    // visible in a screenshot is the selected teacher contact/opening frame.
    await page.waitForFunction(frame => Number(document.body.dataset.frame) === frame, expected, { timeout: 10000 });
    cases.push({ clip: i, text: await page.locator('#dialogue').textContent(), expectedFrame: expected,
      renderedFrame: Number(await page.locator('body').getAttribute('data-frame')) });
    if (i === 2 || i === 3) await page.screenshot({ path: join(root, 'review-case-' + i + '.png'), fullPage: true });
  }
  assert.deepEqual(errors, []);
  await writeFile(join(root, 'review-verification.json'), JSON.stringify({ canvases: 3, cases, errors, seekVerified: true }, null, 2));
  console.log('Verified all six matched clips, audio and rendered seek frames.');
} finally { await browser.close(); }

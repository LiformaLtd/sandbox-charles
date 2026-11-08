import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
const { webkit } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const base = process.env.SCALE_REVIEW_URL;
if (!base) throw Error('Set SCALE_REVIEW_URL to the generated review.html.');
const browser = await webkit.launch(process.env.WEBKIT_EXECUTABLE ? { executablePath: process.env.WEBKIT_EXECUTABLE } : {});
try {
  const page = await browser.newPage({ viewport: { width: 1800, height: 1050 } });
  const errors = [], cases = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(base);
  const ready = () => page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('Ready.'), null, { timeout: 30000 });
  await ready(); assert.equal(await page.locator('canvas').count(), 4);
  assert.equal(await page.locator('#clip option').count(), 16);
  for (let i = 0; i < 16; i++) {
    await page.locator('#clip').selectOption(String(i)); await ready();
    await page.locator('#play').click();
    await page.waitForFunction(() => document.querySelector('#audio').currentTime > .1);
    await page.locator('#play').click();
    const snapshot = page.locator('#snapshots button').nth(1);
    const frame = Number(await snapshot.getAttribute('data-frame'));
    await snapshot.click();
    await page.waitForFunction(f => Number(document.body.dataset.frame) === f, frame);
    cases.push({ index: i, frame, title: await page.locator('#clip option:checked').textContent() });
    if (i === 0 || i === 1 || i === 15) {
      await page.screenshot({ path: `.local/compact-surface-v1/review-case-${i}.png`, fullPage: true });
    }
  }
  assert.deepEqual(errors, []);
  await writeFile('.local/compact-surface-v1/review-verification.json', JSON.stringify({ base, cases, errors }, null, 2));
  console.log('Verified sixteen synchronized cases, audio and exact seek frames.');
} finally { await browser.close(); }

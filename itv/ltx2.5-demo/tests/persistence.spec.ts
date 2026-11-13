import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test.use({ baseURL: async ({}, use) => {
  const url = process.env.TEST_BASE_URL || process.env.LTX_TEST_URL;
  if (!url) throw new Error('The test server did not publish its actual URL.');
  await use(url);
} });

test('restores draft, reconnects to running job, caches video and downloads after refresh', async ({ page }) => {
  let job: { id: string; status: string } | null = null;
  let posts = 0;
  let completed = false;
  const video = await readFile(new URL('./fixtures/clip.mp4', import.meta.url));
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/health') return route.fulfill({ json: { model_ready: true, detail: 'LTX-2.5 Q8 installed locally' } });
    if (route.request().method() === 'POST') {
      posts++;
      const body = route.request().postDataBuffer()!.toString();
      const id = body.match(/name="job_id"\r\n\r\n([^\r]+)/)![1];
      expect(body).toContain('The camera moves slowly around the subject');
      job = { id, status: 'running' };
      return route.fulfill({ status: 202, json: job });
    }
    if (url.pathname.endsWith('/video')) return route.fulfill({ body: video, contentType: 'video/mp4' });
    if (url.pathname.endsWith('/log')) return route.fulfill({ body: 'Stage 1: denoising…' });
    return route.fulfill({ json: { ...job, status: completed ? 'completed' : 'running' } });
  });
  await page.goto('/');
  await page.getByLabel('Starting frame', { exact: true }).setInputFiles(new URL('./fixtures/start.png', import.meta.url).pathname);
  await page.getByLabel('Describe the motion').fill('The camera moves slowly around the subject');
  await expect(page.getByText('Saved on this browser', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByLabel('Describe the motion')).toHaveValue('The camera moves slowly around the subject');
  await expect(page.getByAltText('Starting frame preview')).toBeVisible();
  await page.getByRole('button', { name: 'Generate video' }).click();
  await expect(page.getByText('Generating on your Mac', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByText('Generating on your Mac', { exact: true })).toBeVisible();
  expect(posts).toBe(1);
  completed = true;
  await expect(page.getByRole('link', { name: /Download video/ })).toBeVisible();
  await expect(page.getByText('Video and workspace saved on this browser')).toBeVisible();
  // Verify the MP4 bytes really reached IndexedDB, not just a transient blob URL.
  const storedBytes = await page.evaluate(async () => {
    const db = await new Promise<IDBDatabase>((resolve) => { const r = indexedDB.open('ltx25-demo', 1); r.onsuccess = () => resolve(r.result); });
    return new Promise<number>((resolve) => { const r = db.transaction('records').objectStore('records').get('latestOutput'); r.onsuccess = () => resolve(r.result.blob.size); });
  });
  expect(storedBytes).toBe(video.length);
  await page.reload();
  await expect(page.getByRole('link', { name: /Download video/ })).toBeVisible();
  await expect(page.getByText('Video and workspace restored')).toBeVisible();
  await expect.poll(() => page.locator('video').evaluate((video: HTMLVideoElement) => video.videoWidth)).toBe(32);
  const downloading = page.waitForEvent('download');
  await page.getByRole('link', { name: /Download video/ }).click();
  const download = await downloading;
  expect(download.suggestedFilename()).toMatch(/^ltx25-.*\.mp4$/);
  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(chunk);
  expect(Buffer.concat(chunks)).toEqual(video);
});

test('shows unavailable model without losing the editable saved draft', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: { model_ready: false, detail: 'Model weights are not installed.' } }));
  await page.goto('/');
  await expect(page.getByText('Model weights are not installed.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate video' })).toBeDisabled();
  await page.getByLabel('Describe the motion').fill('Save this while model is offline');
  await expect(page.getByText('Saved on this browser', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByLabel('Describe the motion')).toHaveValue('Save this while model is offline');
});

test('persists random seed and ten seconds, and shows the actual seed for each new job', async ({ page }) => {
  const video = await readFile(new URL('./fixtures/clip.mp4', import.meta.url));
  const seeds = [123456, 987654];
  let job: { id: string; status: string; request: { seed: number; requested_seed: number; frames: number } };
  let posts = 0;
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith('/health')) return route.fulfill({ json: { model_ready: true, detail: 'Ready' } });
    if (route.request().method() === 'POST') {
      const body = route.request().postDataBuffer()!.toString();
      const field = (name: string) => body.match(new RegExp(`name="${name}"\\r\\n\\r\\n([^\\r]+)`))![1];
      expect(field('seed')).toBe('-1');
      expect(field('frames')).toBe('241');
      job = { id: field('job_id'), status: 'completed', request: { seed: seeds[posts++], requested_seed: -1, frames: 241 } };
      return route.fulfill({ status: 202, json: job });
    }
    if (path.endsWith('/video')) return route.fulfill({ body: video, contentType: 'video/mp4' });
    return route.fulfill({ json: job });
  });
  await page.goto('/');
  await expect(page.getByLabel('Seed', { exact: true })).toHaveValue('-1');
  await page.getByLabel('Starting frame', { exact: true }).setInputFiles(new URL('./fixtures/start.png', import.meta.url).pathname);
  await page.getByLabel('Describe the motion').fill('Gentle movement');
  await page.getByLabel('Duration', { exact: true }).selectOption('241');
  await expect(page.getByText('Saved on this browser', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByLabel('Seed', { exact: true })).toHaveValue('-1');
  await expect(page.getByLabel('Duration', { exact: true })).toHaveValue('241');
  await page.getByRole('button', { name: 'Generate video' }).click();
  await expect(page.getByText('Seed used: 123456 · randomly selected')).toBeVisible();
  await expect(page.getByText('Video and workspace saved on this browser')).toBeVisible();
  await page.getByRole('button', { name: 'Generate video' }).click();
  await expect(page.getByText('Seed used: 987654 · randomly selected')).toBeVisible();
  await expect(page.getByText('Video and workspace saved on this browser')).toBeVisible();
  expect(posts).toBe(2);
  await page.reload();
  await expect(page.getByText('Seed used: 987654 · randomly selected')).toBeVisible();
});

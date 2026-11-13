import { test, expect } from '@playwright/test';

test('fast linear fitting saves a separate 36-direction voice without a transcript', async ({ page, request, baseURL }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByText('Running locally', { exact: true })).toBeVisible({ timeout: 45_000 });
  if (await page.locator('.new-recording').getAttribute('open') === null) await page.locator('.new-recording > summary').click();
  await page.getByRole('button', { name: 'Record my voice', exact: true }).click();
  await expect(page.getByText('Listening to you', { exact: true })).toBeVisible();
  await page.waitForTimeout(8000);
  await page.getByRole('button', { name: 'Stop recording' }).click();
  await page.getByLabel('Recording name', { exact: true }).fill('QA linear reference');
  await page.getByRole('checkbox', { name: /This is my own voice/ }).check();
  const creating = page.waitForResponse(r => r.url().includes('/api/voices?') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Save recording & create initial clone', exact: true }).click();
  const source = await (await creating).json();
  const original = await (await request.get(`/api/voices/${source.id}/embedding`)).body();
  let candidateId: string | undefined;
  try {
    await page.getByLabel('New version method', { exact: true }).selectOption('linear');
    await expect(page.getByLabel('The actual words in your recording')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Create linear-fit version', exact: true })).toBeEnabled();
    const fitting = page.waitForResponse(r => r.url().endsWith('/refine') && r.request().method() === 'POST');
    await page.getByRole('button', { name: 'Create linear-fit version', exact: true }).click();
    const response = await fitting;
    expect(response.status()).toBe(200);
    const candidate = await response.json(); candidateId = candidate.id;
    expect(candidate.parentId).toBe(source.id);
    expect(candidate.fit.algorithm).toBe('linear');
    expect(candidate.fit.components).toBe(36);
    expect(candidate.fit.finalError).toBeGreaterThanOrEqual(0);
    await expect(page.getByTestId('fit-error')).toContainText('Linear fitting error:');
    expect(candidate.fit.weights).toHaveLength(37);
    expect(candidate.fit.comparisonSettings.temperature).toBe(0.3);
    await expect(page.getByLabel('Voice version', { exact: true })).toHaveValue(candidate.id);
    await expect(page.getByLabel('Before fit comparison', { exact: true })).toBeVisible();
    await expect(page.getByLabel('After fit comparison', { exact: true })).toBeVisible();
    expect(await (await request.get(`/api/voices/${source.id}/embedding`)).body()).toEqual(original);
    for (const asset of ['original-comparison', 'fitted-comparison']) {
      const clip = await request.get(`/api/voices/${candidate.id}/${asset}`);
      expect(clip.status()).toBe(200); expect((await clip.body()).length).toBeGreaterThan(1000);
    }
    let fitRequestsAfterReload = 0;
    page.on('request', request => { if (request.url().endsWith('/refine') && request.method() === 'POST') fitRequestsAfterReload++; });
    await expect(page.getByLabel('Saved recording')).toHaveValue(source.id);
    await expect(page.getByLabel('Voice version', { exact: true }).locator('option')).toHaveCount(2);
    await page.reload();
    await expect(page.getByLabel('Voice version', { exact: true })).toHaveValue(candidate.id);
    await expect(page.getByLabel('Saved recording')).toHaveValue(source.id);
    await expect(page.locator('.new-recording')).not.toHaveAttribute('open', '');
    expect(fitRequestsAfterReload).toBe(0);
    await page.getByLabel('Voice version', { exact: true }).selectOption(source.id);
    await expect(page.getByRole('button', { name: 'Generate with selected version', exact: true })).toBeEnabled();
    await page.getByLabel('Voice version', { exact: true }).selectOption(candidate.id);
    await expect(page.getByLabel('New version method', { exact: true })).toHaveValue('linear');
    await page.getByRole('button', { name: 'Generate with selected version', exact: true }).click();
    await expect(page.getByLabel('Generated speech', { exact: true })).toBeVisible();
    expect(errors).toEqual([]);
    console.log(JSON.stringify({ linearSeconds: candidate.fit.elapsedSeconds, solveMs: candidate.fit.solveMs, rank: candidate.fit.effectiveRank }));
  } finally {
    for (const id of [candidateId, source.id].filter(Boolean)) {
      expect((await request.delete(`/api/voices/${id}`, { headers: { origin: baseURL! } })).status()).toBe(200);
    }
  }
});

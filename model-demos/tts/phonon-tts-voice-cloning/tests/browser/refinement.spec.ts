import { test, expect } from '@playwright/test';

test('transcribe, cancel and fit a separate voice with the real local models', async ({ page, request, baseURL }) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('/');
  await expect(page.getByText('Running locally', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Record my voice', exact: true }).click();
  await expect(page.getByText('Listening to you', { exact: true })).toBeVisible();
  await page.waitForTimeout(8000);
  await page.getByRole('button', { name: 'Stop recording' }).click();
  await expect(page.getByLabel('Your recorded voice')).toBeVisible();
  await page.getByLabel('Voice name', { exact: true }).fill('QA fitting reference');
  await page.getByRole('checkbox', { name: /This is my own voice/ }).check();
  const create = page.waitForResponse(response => response.url().includes('/api/voices?') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Create my voice', exact: true }).click();
  const source = await (await create).json();
  let candidateId: string | undefined;
  try {
    await page.locator('.fitting-panel > summary').click();
    await page.getByRole('button', { name: 'Transcribe saved recording locally' }).click();
    await expect(page.getByText('Check these words against your recording and correct any mistakes.')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByLabel('The actual words in your recording')).not.toHaveValue('');
    await expect(page.getByRole('button', { name: 'Create fitted voice', exact: true })).toBeDisabled();
    await page.getByRole('checkbox', { name: /I checked this transcript/ }).check();
    await page.getByLabel('Fitting focus').selectOption('pronunciation');
    const cancelling = page.waitForResponse(response => response.url().endsWith('/refine') && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Create fitted voice', exact: true }).click();
    await page.getByRole('button', { name: 'Cancel fitting', exact: true }).click({ timeout: 15_000 });
    expect((await (await cancelling).json()).cancelled).toBe(true);
    await expect(page.getByText('Voice fitting cancelled. Your original voice is unchanged.')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Create fitted voice', exact: true })).toBeEnabled();
    const fitting = page.waitForResponse(response => response.url().endsWith('/refine') && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Create fitted voice', exact: true }).click();
    const response = await fitting;
    expect(response.status()).toBe(200);
    const candidate = await response.json(); candidateId = candidate.id;
    expect(candidate.parentId).toBe(source.id);
    expect(candidate.fit.focus).toBe('pronunciation');
    expect(candidate.fit.after.loss).toBeLessThan(candidate.fit.before.loss);
    await expect(page.getByLabel('Your voices')).toHaveValue(candidate.id);
    await expect(page.getByLabel('Original clone comparison', { exact: true })).toBeVisible();
    await expect(page.getByLabel('Fitted clone comparison', { exact: true })).toBeVisible();
    await page.screenshot({ path: '.data/testing/fitting.png', fullPage: true });
    expect(errors).toEqual([]);
    console.log(JSON.stringify({ fitSeconds: candidate.fit.elapsedSeconds,
      before: candidate.fit.heldOutBefore.speakerCosine, after: candidate.fit.heldOutAfter.speakerCosine }));
  } finally {
    for (const id of [candidateId, source.id].filter(Boolean)) {
      const response = await request.delete(`/api/voices/${id}`, { headers: { origin: baseURL! } });
      expect(response.status()).toBe(200);
    }
  }
});

import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const health = { ready: false, min_image_side: 256, max_upload_bytes: 10485760, disclaimer: 'Model score for one Stable Diffusion v1.5 dataset; not proof of authenticity; may be wrong on other generators, screenshots or compressed images.' };

test('desktop and mobile workspace remain accessible without a model', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: health }));
  await page.goto('/');
  await expect(page.getByText('Research model not connected')).toBeVisible();
  await page.screenshot({ path: 'test-results/workspace-desktop.png', fullPage: true });
  expect((await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()).violations).toEqual([]);
  await page.getByRole('button', { name: 'About the study' }).click();
  await expect(page.getByRole('heading', { name: 'Evidence in the frequency domain.' })).toBeVisible();
  await page.getByRole('button', { name: 'Analyze an image' }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => document.activeElement?.blur());
  await page.screenshot({ path: 'test-results/workspace-mobile.png', fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect((await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()).violations).toEqual([]);
  await page.getByRole('button', { name: 'Choose an image' }).focus();
  const opening = page.waitForEvent('filechooser');
  await page.keyboard.press('Enter');
  await (await opening).setFiles([]);
});

test('analysis result, removal, and errors are usable with mocked synthetic responses', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: { ...health, ready: true } }));
  // Synthetic UI response only; no scientific prediction is claimed by this test.
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a1ioAAAAASUVORK5CYII=', 'base64');
  await page.route('**/api/analyze', route => route.fulfill({ json: { label: 'AI-generated', ai_probability: .72, warnings: [], spectrum_png: `data:image/png;base64,${png.toString('base64')}`, rapsd: { radius: [1, 64, 128], log_power: [3, 2, 1] } } }));
  await page.goto('/');
  await page.getByLabel('Choose an image').setInputFiles({ name: 'synthetic-fixture.png', mimeType: 'image/png', buffer: png });
  await page.getByRole('button', { name: 'Analyze image' }).click();
  await expect(page.getByRole('heading', { name: 'AI-generated', exact: true })).toBeVisible();
  await expect(page.getByRole('meter')).toHaveAttribute('aria-valuenow', '72');
  expect((await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()).violations).toEqual([]);
  await page.getByRole('button', { name: 'Remove image' }).click();
  await expect(page.getByRole('meter')).toHaveCount(0);
  await page.route('**/api/analyze', route => route.fulfill({ status: 422, json: { detail: 'Image smaller than target: centered crop side 1 < 256' } }));
  await page.getByLabel('Choose an image').setInputFiles({ name: 'tiny.png', mimeType: 'image/png', buffer: png });
  await page.getByRole('button', { name: 'Analyze image' }).click();
  await expect(page.getByRole('alert')).toContainText('Image smaller than target');
});

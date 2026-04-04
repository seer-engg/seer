/**
 * Playwright Global Setup for Authentication
 *
 * Runs once before all tests to:
 * 1. Log in via Clerk's sign-in UI
 * 2. Save the authenticated session state
 * 3. All tests then load this state to start authenticated
 *
 * Requires E2E_CLERK_EMAIL and E2E_CLERK_PASSWORD env vars.
 */
import { chromium, type FullConfig } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const AUTH_FILE = path.join(__dirname, '.auth/user.json');

async function globalSetup(config: FullConfig) {
  const { baseURL } = config.projects[0].use;

  const email = process.env.E2E_CLERK_EMAIL;
  const password = process.env.E2E_CLERK_PASSWORD;

  if (!email || !password) {
    console.error('❌ E2E test credentials not configured');
    console.error('   Set E2E_CLERK_EMAIL and E2E_CLERK_PASSWORD in .env.test');
    throw new Error('Missing E2E test credentials');
  }

  console.log('🔐 Logging in for E2E tests...');

  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    // Step 1: Navigate to sign-in page
    await page.goto(`${baseURL}/sign-in`);

    // Wait for Clerk to render the email field
    const emailInput = page.getByRole('textbox', { name: 'Email address' });
    await emailInput.waitFor({ state: 'visible', timeout: 15000 });

    // Dismiss PostHog survey if it appears (overlays the form)
    const surveyClose = page.getByRole('button', { name: 'Close survey' });
    if (await surveyClose.isVisible({ timeout: 2000 }).catch(() => false)) {
      await surveyClose.click();
    }

    // Fill email and submit
    await emailInput.fill(email);
    await page.getByRole('button', { name: 'Continue', exact: true }).click();

    // Step 2: Clerk navigates to factor-one for password
    const passwordInput = page.getByRole('textbox', { name: 'Password' });
    await passwordInput.waitFor({ state: 'visible', timeout: 15000 });
    await passwordInput.fill(password);
    await page.getByRole('button', { name: 'Continue', exact: true }).click();

    // Step 3: Wait for authenticated redirect (dashboard or home)
    await page.waitForURL((url) => !url.pathname.includes('/sign-in'), {
      timeout: 30000,
    });

    console.log('✅ Login successful');

    // Save authentication state (cookies, localStorage)
    await context.storageState({ path: AUTH_FILE });
    console.log(`💾 Auth state saved to ${AUTH_FILE}`);
  } catch (error) {
    console.error('❌ Login failed:', error);
    await page.screenshot({ path: 'e2e/.auth/login-failure.png' });
    throw error;
  } finally {
    await browser.close();
  }
}

export default globalSetup;

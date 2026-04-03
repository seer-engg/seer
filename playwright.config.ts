import { defineConfig, devices } from '@playwright/test';
import { config } from 'dotenv';

// Load test environment first (for E2E credentials), then main .env
config({ path: '.env.test', quiet: true });
config({ quiet: true });

const port = process.env.VITE_DEV_PORT || '5173';
const baseURL = `http://localhost:${port}`;

export default defineConfig({
  testDir: './e2e/tests',

  // Run global setup to authenticate before all tests
  globalSetup: './e2e/global-setup.ts',

  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: [
    ['html'],
    ['list'],
    ['json', { outputFile: 'test-results/playwright-results.json' }],
  ],
  use: {
    baseURL,
    // Load saved authentication state for all tests
    storageState: './e2e/.auth/user.json',
    trace: 'on',
    screenshot: 'on',
    video: 'on',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});

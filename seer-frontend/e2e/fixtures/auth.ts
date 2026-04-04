/**
 * Authentication Fixture for E2E Tests
 *
 * Authentication is now handled by:
 * 1. global-setup.ts - Logs in once and saves session state
 * 2. playwright.config.ts - Loads storageState for all tests
 *
 * This fixture simply re-exports the base test and expect.
 * All tests using this fixture will automatically be authenticated.
 */
import { test as base, expect } from '@playwright/test';

export const test = base;
export { expect };

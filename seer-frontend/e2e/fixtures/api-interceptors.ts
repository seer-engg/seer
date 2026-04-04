/**
 * API Interceptors for E2E Tests
 *
 * Reusable route handlers for mocking API responses in Playwright tests.
 * These helpers enable testing error handling, conflict resolution, and
 * edge cases that are difficult to trigger with real API calls.
 */
import type { Page, Route } from '@playwright/test';

/**
 * Mock a 409 Conflict response on workflow draft saves.
 * Simulates concurrent editing scenarios where another session modified the workflow.
 */
export async function mock409Conflict(page: Page): Promise<void> {
  await page.route('**/api/v1/workflows/*/draft', async (route: Route) => {
    if (route.request().method() === 'PATCH') {
      await route.fulfill({
        status: 409,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Workflow was modified by another session',
          type: 'conflict'
        })
      });
    } else {
      await route.continue();
    }
  });
}

/**
 * Mock a network error (connection failure) on workflow draft saves.
 * Simulates network disconnection or server unavailability.
 */
export async function mockNetworkError(page: Page): Promise<void> {
  await page.route('**/api/v1/workflows/*/draft', (route: Route) => route.abort('failed'));
}

/**
 * Mock a network timeout on workflow draft saves.
 * Useful for testing loading states and timeout handling.
 * @param delayMs - How long to delay before timing out (default: never resolves)
 */
export async function mockNetworkTimeout(page: Page, delayMs?: number): Promise<void> {
  await page.route('**/api/v1/workflows/*/draft', async (route: Route) => {
    if (route.request().method() === 'PATCH') {
      if (delayMs) {
        await new Promise(resolve => setTimeout(resolve, delayMs));
      } else {
        // Never resolve - will cause timeout
        await new Promise(() => {});
      }
    } else {
      await route.continue();
    }
  });
}

/**
 * Mock a 422 Validation Error with node-specific error details.
 * Simulates backend validation failures for specific nodes.
 */
export async function mockValidationError(
  page: Page,
  nodeId: string,
  message: string,
  code: string = 'VALIDATION_ERROR'
): Promise<void> {
  await page.route('**/api/v1/workflows/*/draft', async (route: Route) => {
    if (route.request().method() === 'PATCH') {
      await route.fulfill({
        status: 422,
        contentType: 'application/problem+json',
        body: JSON.stringify({
          type: 'validation_error',
          title: 'Validation Error',
          node_errors: {
            [nodeId]: {
              code,
              message
            }
          }
        })
      });
    } else {
      await route.continue();
    }
  });
}

/**
 * Mock a 402 Payment Required response.
 * Simulates subscription/payment wall scenarios.
 */
export async function mock402PaymentRequired(page: Page): Promise<void> {
  await page.route('**/api/v1/workflows/*/draft', async (route: Route) => {
    if (route.request().method() === 'PATCH') {
      await route.fulfill({
        status: 402,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Subscription required',
          type: 'payment_required',
          upgrade_url: '/settings/billing'
        })
      });
    } else {
      await route.continue();
    }
  });
}

/**
 * Mock a 500 Internal Server Error.
 * Simulates unexpected server-side failures.
 */
export async function mock500ServerError(page: Page): Promise<void> {
  await page.route('**/api/v1/workflows/*/draft', async (route: Route) => {
    if (route.request().method() === 'PATCH') {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Internal server error',
          type: 'server_error'
        })
      });
    } else {
      await route.continue();
    }
  });
}

/**
 * Count API calls matching a URL pattern.
 * Returns a function that returns the current count when called.
 * Pattern example: "**\/api/v1/workflows/*\/draft" (PATCH calls)
 */
export async function countApiCalls(
  page: Page,
  pattern: string,
  method?: string
): Promise<() => number> {
  let count = 0;

  await page.route(pattern, async (route: Route) => {
    if (!method || route.request().method() === method) {
      count++;
    }
    await route.continue();
  });

  return () => count;
}

/**
 * Capture API request bodies for inspection.
 * Returns a function that returns all captured request bodies.
 * Useful for verifying what data is being sent in save requests.
 */
export async function captureApiRequests(
  page: Page,
  pattern: string,
  method?: string
): Promise<() => unknown[]> {
  const bodies: unknown[] = [];

  await page.route(pattern, async (route: Route) => {
    const request = route.request();
    if (!method || request.method() === method) {
      try {
        const body = request.postDataJSON();
        bodies.push(body);
      } catch {
        bodies.push(request.postData());
      }
    }
    await route.continue();
  });

  return () => [...bodies];
}

/**
 * Mock API to succeed after N failures.
 * Useful for testing retry logic.
 *
 * @param failCount - Number of times to return failure before succeeding
 * @param errorStatus - HTTP status code for failures (default: 500)
 */
export async function mockFailThenSucceed(
  page: Page,
  pattern: string,
  failCount: number,
  errorStatus: number = 500
): Promise<void> {
  let attempts = 0;

  await page.route(pattern, async (route: Route) => {
    attempts++;
    if (attempts <= failCount) {
      await route.fulfill({
        status: errorStatus,
        contentType: 'application/json',
        body: JSON.stringify({ detail: `Failure ${attempts} of ${failCount}` })
      });
    } else {
      await route.continue();
    }
  });
}

/**
 * Remove all route handlers for a pattern.
 * Call this to restore normal API behavior after mocking.
 */
export async function clearMocks(page: Page, pattern: string): Promise<void> {
  await page.unroute(pattern);
}

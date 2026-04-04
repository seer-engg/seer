---
name: e2eTests
description:  use when creating or updating end-to-end tests for the Seer application. This skill provides guidance on how to structure and implement e2e tests effectively.
---


## E2E Tests

We use actual testing backend for our E2E playwright test with all auth config connected.
and hence it is hard to guess the rendered tool/trigger/node configs and other data , since it comes via apis.

To understand the app better you can use  playwright mcp to open our localy runnig frontend at  http://172.17.0.1:3003
and then you can inspect the actual app and design your playwright tests by combining the observations from actual app and the codebase.Credintials for login are at .env.test file in the repo.


1. **Use semantic locators only** — always use `getByRole`, `getByLabel`, `getByText`, or `getByTestId`; never use CSS selectors, XPath, or positional selectors like `nth-child`.

2. **Never use `waitForTimeout`** — instead wait for specific UI state using `await expect(locator).toBeVisible()` or `await page.waitForResponse('**/api/...')` before proceeding.

3. **Always use web-first assertions** — use `await expect(locator).toBeVisible/toHaveText/toBeEnabled()` so Playwright auto-retries; never read DOM values and assert separately.

4. **Assert UI is ready before acting** — before every meaningful interaction (click, fill, submit), assert the target element is visible/enabled first to prevent race conditions with your real backend's response times.
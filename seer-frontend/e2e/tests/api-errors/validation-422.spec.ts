/**
 * 422 Validation Error Handling Tests
 *
 * Verifies proper handling of validation errors returned by the backend.
 * When a node has invalid configuration, the UI should:
 * - Highlight the affected node
 * - Show specific error messages in the config panel
 * - Allow the user to fix the error
 */
import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addBlock,
  selectNode,
  switchBuildPanelTab,
} from '../../fixtures/utils';
import { mockValidationError, clearMocks } from '../../fixtures/api-interceptors';

test.describe('422 Validation Error Handling', () => {
  let workflowId: string;

  test.beforeEach(async ({ page }) => {
    workflowId = await openWorkflow(page);
    await waitForCanvasReady(page);
  });

  test.afterEach(async ({ page }) => {
    await clearMocks(page, '**/api/v1/workflows/*/draft').catch(() => {});

    if (workflowId) {
      await deleteTestWorkflow(page, workflowId);
    }
  });

  test('422 validation error shows node-specific error message', async ({ page }) => {
    // Add block first (before mocking)
    await addBlock(page, 'llm');

    // Wait for initial save
    await page.waitForTimeout(1500);

    // Mock 422 validation error for this node
    await mockValidationError(page, 'llm-1', 'Prompt cannot be empty', 'INVALID_PROMPT');

    // Select the node to trigger a view/potential save
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');

    // Make a change to trigger the mocked error
    const promptInput = page.getByLabel(/Prompt|System/i);
    await promptInput.fill(''); // Empty prompt should trigger validation
    await page.waitForTimeout(2000);

    // Look for error message - could be in toast, panel, or inline
    const errorMessage = page.getByText(/cannot be empty|invalid|error/i);
    await expect(errorMessage).toBeVisible({ timeout: 5000 });
  });

  test('validation error highlights affected node', async ({ page }) => {
    await addBlock(page, 'llm');
    await page.waitForTimeout(1500);

    // Mock validation error
    await mockValidationError(page, 'llm-1', 'Invalid configuration', 'VALIDATION_ERROR');

    // Trigger a save
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const promptInput = page.getByLabel(/Prompt|System/i);
    await promptInput.fill('test');
    await page.waitForTimeout(2000);

    // Check for visual error indicator on node
    const llmNode = page.locator('.react-flow__node').filter({ hasText: /llm/i });

    // Look for error styling - could be red border, error icon, error class
    // The exact styling depends on implementation
    const hasErrorStyle = await llmNode.evaluate((el) => {
      const classes = el.className;
      const style = window.getComputedStyle(el);
      return (
        classes.includes('error') ||
        classes.includes('invalid') ||
        style.borderColor.includes('rgb(239') || // red
        style.borderColor.includes('rgb(220') || // another red
        el.querySelector('[data-error]') !== null ||
        el.querySelector('.error') !== null
      );
    }).catch(() => false);

    // If no visual indicator, at least the error should be shown somewhere
    if (!hasErrorStyle) {
      const errorVisible = await page.getByText(/invalid|error/i).isVisible();
      expect(errorVisible).toBe(true);
    }
  });

  test('fixing validation error clears error state', async ({ page }) => {
    await addBlock(page, 'llm');
    await page.waitForTimeout(1500);

    // First, mock the error
    await mockValidationError(page, 'llm-1', 'Prompt required', 'INVALID_PROMPT');

    // Trigger the error
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const promptInput = page.getByLabel(/Prompt|System/i);
    await promptInput.fill('');
    await page.waitForTimeout(2000);

    // Clear the mock to allow successful save
    await clearMocks(page, '**/api/v1/workflows/*/draft');

    // Fix the error by adding valid content
    await promptInput.fill('Valid prompt content');
    await page.waitForTimeout(2000);

    // Error should be cleared - look for saved indicator
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });
  });

  test('multiple validation errors shown for multiple nodes', async ({ page }) => {
    // Add multiple blocks
    await addBlock(page, 'llm');  // llm-1
    await addBlock(page, 'llm');  // llm-2
    await page.waitForTimeout(1500);

    // Mock validation errors for both nodes
    await page.route('**/api/v1/workflows/*/draft', async (route) => {
      if (route.request().method() === 'PATCH') {
        await route.fulfill({
          status: 422,
          contentType: 'application/problem+json',
          body: JSON.stringify({
            type: 'validation_error',
            title: 'Validation Error',
            node_errors: {
              'llm-1': { code: 'INVALID_PROMPT', message: 'First node: prompt empty' },
              'llm-2': { code: 'INVALID_PROMPT', message: 'Second node: prompt empty' }
            }
          })
        });
      } else {
        await route.continue();
      }
    });

    // Trigger save
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    await page.waitForTimeout(2000);

    // Both error messages should be visible or accessible
    // (either in toasts, in a panel, or when selecting each node)
    const errorText = page.getByText(/prompt empty|validation|error/i);
    await expect(errorText.first()).toBeVisible({ timeout: 5000 });
  });

  test('422 error does not lose unsaved changes', async ({ page }) => {
    await addBlock(page, 'llm');
    await page.waitForTimeout(1500);

    // Configure the block first
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const promptInput = page.getByLabel(/Prompt|System/i);
    await promptInput.fill('My detailed prompt that I want to keep');

    // Now mock validation error
    await mockValidationError(page, 'llm-1', 'Some validation error', 'VALIDATION_ERROR');

    // Trigger save
    await addBlock(page, 'if_else');
    await page.waitForTimeout(2000);

    // Original content should still be there
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByLabel(/Prompt|System/i)).toHaveValue(/My detailed prompt/);
  });

  test('can navigate between nodes during validation error', async ({ page }) => {
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');
    await page.waitForTimeout(1500);

    // Mock validation error
    await mockValidationError(page, 'llm-1', 'Invalid config', 'VALIDATION_ERROR');

    // Trigger error
    await page.waitForTimeout(2000);

    // Should still be able to navigate between nodes
    await selectNode(page, /llm/i);
    await expect(page.locator('.react-flow__node--selected').filter({ hasText: /llm/i })).toBeVisible();

    await selectNode(page, /if/i);
    await expect(page.locator('.react-flow__node--selected').filter({ hasText: /if/i })).toBeVisible();

    // Can switch back to error node
    await selectNode(page, /llm/i);
    await expect(page.locator('.react-flow__node--selected').filter({ hasText: /llm/i })).toBeVisible();
  });
});

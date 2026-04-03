/**
 * 409 Conflict Error Handling Tests
 *
 * Verifies proper handling of concurrent modification conflicts.
 * When another session modifies the workflow, the UI should:
 * - Show an error toast/notification
 * - Preserve local (unsaved) changes
 * - Allow the user to resolve the conflict
 */
import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addBlock,
  configureLLMBlock,
  selectNode,
  switchBuildPanelTab,
} from '../../fixtures/utils';
import { mock409Conflict, clearMocks } from '../../fixtures/api-interceptors';

test.describe('409 Conflict Error Handling', () => {
  let workflowId: string;

  test.beforeEach(async ({ page }) => {
    workflowId = await openWorkflow(page);
    await waitForCanvasReady(page);
  });

  test.afterEach(async ({ page }) => {
    // Clear any mocks before cleanup
    await clearMocks(page, '**/api/v1/workflows/*/draft').catch(() => {});

    if (workflowId) {
      await deleteTestWorkflow(page, workflowId);
    }
  });

  test('409 conflict shows error toast and preserves local state', async ({ page }) => {
    // First, add a block normally (before mocking)
    await addBlock(page, 'llm');
    await configureLLMBlock(page, /llm/i, 'My important prompt');

    // Wait for initial save to complete
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Now mock 409 response on subsequent saves
    await mock409Conflict(page);

    // Trigger another save by adding a block
    await addBlock(page, 'if_else');
    await page.waitForTimeout(2000); // Wait for debounced save attempt

    // Verify error toast/notification appears
    // Look for various possible error message patterns
    const errorIndicator = page.getByText(/conflict|modified|error|failed/i);
    await expect(errorIndicator).toBeVisible({ timeout: 5000 });

    // Verify local state is preserved - both blocks should still be visible
    const nodeCount = await page.locator('.react-flow__node').count();
    expect(nodeCount).toBe(2);

    // Verify original config still in UI
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const promptField = page.getByLabel(/Prompt|System/i);
    await expect(promptField).toHaveValue(/My important prompt/);
  });

  test('409 conflict shows unsaved indicator', async ({ page }) => {
    await addBlock(page, 'llm');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock 409 conflict
    await mock409Conflict(page);

    // Make a change that will fail to save
    await configureLLMBlock(page, /llm/i, 'Change that cannot save');
    await page.waitForTimeout(2000);

    // Should show some indication of unsaved/error state
    // This could be "Unsaved", "Error", "Retry", etc.
    const unsavedIndicator = page.getByText(/unsaved|error|retry|conflict/i);
    await expect(unsavedIndicator).toBeVisible({ timeout: 5000 });
  });

  test('can recover from 409 by refreshing page', async ({ page }) => {
    // Add initial content
    await addBlock(page, 'llm');
    await configureLLMBlock(page, /llm/i, 'Original saved content');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock 409 and try to add more content
    await mock409Conflict(page);
    await addBlock(page, 'if_else');
    await page.waitForTimeout(2000);

    // Clear mock before refresh
    await clearMocks(page, '**/api/v1/workflows/*/draft');

    // Refresh should load last saved state from server
    await page.reload();
    await waitForCanvasReady(page);

    // Original LLM should exist with its config
    await expect(page.locator('.react-flow__node').filter({ hasText: /llm/i })).toBeVisible();
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByLabel(/Prompt|System/i)).toHaveValue(/Original saved/);

    // The if_else block that failed to save may or may not exist
    // depending on whether the conflict was for that specific change
  });

  test('409 does not prevent viewing existing content', async ({ page }) => {
    // Add blocks first
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock 409
    await mock409Conflict(page);

    // Should still be able to select and view nodes
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const configPanel = page.locator('[data-testid="build-panel"]');
    await expect(configPanel).toBeVisible();

    await selectNode(page, /if/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(configPanel).toBeVisible();
  });

  test('multiple 409 errors do not crash the app', async ({ page }) => {
    await addBlock(page, 'llm');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock 409
    await mock409Conflict(page);

    // Make multiple changes that will all fail
    for (let i = 0; i < 3; i++) {
      await selectNode(page, /llm/i);
      await switchBuildPanelTab(page, 'Config');
      const promptInput = page.getByLabel(/Prompt|System/i);
      await promptInput.fill(`Attempt ${i + 1}`);
      await page.waitForTimeout(1500);
    }

    // App should still be functional
    const nodeCount = await page.locator('.react-flow__node').count();
    expect(nodeCount).toBe(1);

    // Canvas should still be interactive
    const canvas = page.locator('[data-testid="workflow-canvas"]');
    await expect(canvas).toBeVisible();
  });
});

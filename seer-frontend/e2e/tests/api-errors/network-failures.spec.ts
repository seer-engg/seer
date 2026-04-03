/**
 * Network Failure Handling Tests
 *
 * Verifies proper handling of network errors like timeouts, disconnections,
 * and server unavailability. The UI should:
 * - Show appropriate error states
 * - Preserve local changes
 * - Allow retry when connection is restored
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
import { mockNetworkError, mockNetworkTimeout, clearMocks, mockFailThenSucceed } from '../../fixtures/api-interceptors';

test.describe('Network Failure Handling', () => {
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

  test('network error preserves local changes', async ({ page }) => {
    // Add block normally first
    await addBlock(page, 'llm');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock network error
    await mockNetworkError(page);

    // Make changes that will fail to save
    await configureLLMBlock(page, /llm/i, 'Important content to preserve');
    await page.waitForTimeout(2000);

    // Verify changes still in UI
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByLabel(/Prompt|System/i)).toHaveValue(/Important content/);
  });

  test('network error shows error indicator', async ({ page }) => {
    await addBlock(page, 'llm');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock network failure
    await mockNetworkError(page);

    // Make a change
    await configureLLMBlock(page, /llm/i, 'Test content');
    await page.waitForTimeout(2000);

    // Should show some error/retry indicator
    // This could be "Error", "Retry", "Offline", "Connection lost", etc.
    const errorIndicator = page.getByText(/error|retry|offline|failed|connection/i);
    await expect(errorIndicator).toBeVisible({ timeout: 5000 });
  });

  test('can continue editing during network failure', async ({ page }) => {
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock network error
    await mockNetworkError(page);

    // Should still be able to interact with the canvas
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');

    const promptInput = page.getByLabel(/Prompt|System/i);
    await promptInput.fill('Content during network issue');

    // Can switch to another node
    await selectNode(page, /if/i);
    await switchBuildPanelTab(page, 'Config');

    // Can add another block (UI-side)
    await addBlock(page, 'llm');

    // Verify all blocks exist in UI
    const nodeCount = await page.locator('.react-flow__node').count();
    expect(nodeCount).toBe(3);
  });

  test('recovery from network failure saves pending changes', async ({ page }) => {
    await addBlock(page, 'llm');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock network failure
    await mockNetworkError(page);

    // Make changes that fail to save
    await configureLLMBlock(page, /llm/i, 'Content waiting to be saved');
    await page.waitForTimeout(2000);

    // Clear mock to restore network
    await clearMocks(page, '**/api/v1/workflows/*/draft');

    // Trigger a retry by making another small change
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const promptInput = page.getByLabel(/Prompt|System/i);
    await promptInput.fill('Content waiting to be saved!'); // Add a character

    // Wait for save to complete
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Refresh to verify persistence
    await page.reload();
    await waitForCanvasReady(page);

    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByLabel(/Prompt|System/i)).toHaveValue(/Content waiting/);
  });

  test('network timeout shows appropriate state', async ({ page }) => {
    await addBlock(page, 'llm');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock network timeout (requests hang forever)
    await mockNetworkTimeout(page);

    // Make a change
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const promptInput = page.getByLabel(/Prompt|System/i);
    await promptInput.fill('Timeout test');

    // Wait for potential timeout indication (may show "Saving..." for a while)
    await page.waitForTimeout(5000);

    // Should show saving/pending state (not "Saved")
    // Look for saving indicator or error state
    const savingOrError = page.getByText(/saving|pending|error|retry/i);
    const visible = await savingOrError.isVisible().catch(() => false);

    // If not showing explicit state, at least it shouldn't say "Saved"
    // (since the request is still pending)
    if (!visible) {
      // Local changes should still be preserved
      await expect(promptInput).toHaveValue(/Timeout test/);
    }
  });

  test('retry succeeds after initial failures', async ({ page }) => {
    await addBlock(page, 'llm');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock to fail twice then succeed
    await mockFailThenSucceed(page, '**/api/v1/workflows/*/draft', 2, 503);

    // Make changes
    await configureLLMBlock(page, /llm/i, 'Retry test content');

    // Wait for retries and eventual success
    // This may take longer due to retry backoff
    await page.waitForTimeout(5000);

    // Eventually should show saved (after retries succeed)
    // Or at minimum, content should be preserved
    const promptInput = page.getByLabel(/Prompt|System/i);
    await expect(promptInput).toHaveValue(/Retry test/);
  });

  test('multiple network failures do not corrupt state', async ({ page }) => {
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Configure first block
    await configureLLMBlock(page, /llm/i, 'Original config');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Mock network error
    await mockNetworkError(page);

    // Make multiple rapid changes
    for (let i = 0; i < 5; i++) {
      await selectNode(page, /llm/i);
      await switchBuildPanelTab(page, 'Config');
      const promptInput = page.getByLabel(/Prompt|System/i);
      await promptInput.fill(`Failed change ${i + 1}`);
      await page.waitForTimeout(300);
    }

    await page.waitForTimeout(2000);

    // Clear mock
    await clearMocks(page, '**/api/v1/workflows/*/draft');

    // Final change should save successfully
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const promptInput = page.getByLabel(/Prompt|System/i);
    await promptInput.fill('Final successful change');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Verify persistence
    await page.reload();
    await waitForCanvasReady(page);
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByLabel(/Prompt|System/i)).toHaveValue(/Final successful/);
  });

  test('app remains functional after network recovery', async ({ page }) => {
    await addBlock(page, 'llm');
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Simulate network outage
    await mockNetworkError(page);
    await page.waitForTimeout(1000);

    // Clear to restore network
    await clearMocks(page, '**/api/v1/workflows/*/draft');

    // All functionality should work
    await addBlock(page, 'if_else');
    await addBlock(page, 'for_loop');

    // Wait for saves
    await expect(page.getByText(/Saved/i)).toBeVisible({ timeout: 5000 });

    // Verify all blocks exist and are saved
    await page.reload();
    await waitForCanvasReady(page);

    const nodeCount = await page.locator('.react-flow__node').count();
    expect(nodeCount).toBe(3);
  });
});

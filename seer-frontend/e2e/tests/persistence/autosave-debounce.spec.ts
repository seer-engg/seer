/**
 * Autosave Debounce Tests
 *
 * Verifies that the 1000ms debounce mechanism works correctly,
 * preventing excessive API calls during rapid editing while still
 * ensuring changes are eventually saved.
 */
import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  waitForWorkflowLoaded,
  waitForAutosave,
  addBlock,
  selectNode,
  switchBuildPanelTab,
} from '../../fixtures/utils';

test.describe('Autosave Debounce', () => {
  let workflowId: string;

  test.beforeEach(async ({ page }) => {
    workflowId = await openWorkflow(page);
    await waitForCanvasReady(page);
  });

  test.afterEach(async ({ page }) => {
    if (workflowId) {
      await deleteTestWorkflow(page, workflowId);
    }
  });

  test('rapid edits trigger single save call (debounce works)', async ({ page }) => {
    let saveCallCount = 0;

    // Intercept save calls to count them
    await page.route('**/api/v1/workflows/*/draft', async (route) => {
      if (route.request().method() === 'PATCH') {
        saveCallCount++;
      }
      await route.continue();
    });

    // Add an LLM block
    await addBlock(page, 'llm');
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');

    // Find the prompt input field
    const promptInput = page.getByRole('textbox', { name: /Prompt/i });
    await promptInput.waitFor({ state: 'visible', timeout: 5000 });

    // Reset counter after initial setup
    saveCallCount = 0;

    // Make 5 rapid edits within 500ms (faster than debounce)
    await promptInput.fill('A');
    await page.waitForTimeout(100);
    await promptInput.fill('AB');
    await page.waitForTimeout(100);
    await promptInput.fill('ABC');
    await page.waitForTimeout(100);
    await promptInput.fill('ABCD');
    await page.waitForTimeout(100);
    await promptInput.fill('ABCDE');

    // Wait for debounce (1000ms) + buffer for save to complete
    await page.waitForTimeout(2000);

    // Should have only 1-2 save calls, not 5
    // (1 for debounced batch, possibly 1 more if timing edge case)
    expect(saveCallCount).toBeLessThanOrEqual(2);
    expect(saveCallCount).toBeGreaterThanOrEqual(1);
  });

  test('slow sequential edits trigger individual saves', async ({ page }) => {
    let saveCallCount = 0;

    // Intercept save calls
    await page.route('**/api/v1/workflows/*/draft', async (route) => {
      if (route.request().method() === 'PATCH') {
        saveCallCount++;
      }
      await route.continue();
    });

    await addBlock(page, 'llm');
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');

    const promptInput = page.getByRole('textbox', { name: /Prompt/i });
    await promptInput.waitFor({ state: 'visible', timeout: 5000 });

    saveCallCount = 0;

    // Make 3 edits with enough time between them for debounce to fire
    await promptInput.fill('First edit');
    await page.waitForTimeout(1500); // Wait for debounce + save

    await promptInput.fill('Second edit');
    await page.waitForTimeout(1500);

    await promptInput.fill('Third edit');
    await page.waitForTimeout(1500);

    // Should have roughly 3 save calls (one per edit)
    // Allow some variance for timing
    expect(saveCallCount).toBeGreaterThanOrEqual(2);
  });

  test('adding multiple blocks saves all', async ({ page }) => {
    test.setTimeout(90000); // Multiple blocks with debounce waits

    // Add multiple blocks (addBlock has built-in debounce wait)
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');
    await addBlock(page, 'llm');

    // Wait for final autosave
    await waitForAutosave(page);

    // Verify all 3 blocks exist
    const nodeCount = await page.locator('.react-flow__node').count();
    expect(nodeCount).toBe(3);

    // Reload to verify persistence
    await page.reload();
    await waitForWorkflowLoaded(page, 3); // Expect 3 blocks to load

    // All blocks should still be there
    const persistedNodeCount = await page.locator('.react-flow__node').count();
    expect(persistedNodeCount).toBe(3);
  });

  test('changes persist after debounce completes', async ({ page }) => {
    await addBlock(page, 'llm');
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');

    const promptInput = page.getByRole('textbox', { name: /Prompt/i });
    await promptInput.waitFor({ state: 'visible', timeout: 5000 });

    // Make an edit
    await promptInput.fill('Testing persistence');

    // Wait for debounce + save
    await page.waitForTimeout(2500);

    // Reload to verify persistence
    await page.reload();
    await waitForWorkflowLoaded(page, 1); // Expect 1 LLM block

    // Verify the change was saved
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const savedPrompt = page.getByRole('textbox', { name: /Prompt/i });
    await expect(savedPrompt).toHaveValue(/Testing persistence/);
  });
});

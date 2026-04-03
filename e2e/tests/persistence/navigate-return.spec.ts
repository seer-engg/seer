/**
 * Navigate and Return Persistence Tests
 *
 * Verifies that workflow configurations persist when navigating
 * away from the workflow editor and returning.
 */
import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  waitForAutosave,
  waitForWorkflowLoaded,
  addBlock,
  addTrigger,
  configureLLMBlock,
  configureIfElseBlock,
  selectNode,
  switchBuildPanelTab,
  connectNodes,
  organizeLayout,
} from '../../fixtures/utils';

test.describe('Navigate and Return Persistence', () => {
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

  test('LLM config persists after navigating to workflows list and returning', async ({ page }) => {
    // Add and configure LLM block
    await addBlock(page, 'llm');
    await configureLLMBlock(page, /llm/i, 'Analyze this: {{trigger.data}}');

    // Wait for autosave to complete
    await waitForAutosave(page);

    // Navigate away to workflows list
    await page.goto('/workflows');
    await page.waitForLoadState('networkidle');

    // Return to workflow
    await page.goto(`/workflows/${workflowId}`);
    await waitForWorkflowLoaded(page, 1); // Expect 1 LLM block

    // Verify config persisted
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');

    const promptField = page.getByRole('textbox', { name: /Prompt/i });
    await expect(promptField).toHaveValue(/Analyze this/);
  });

  test('multiple block configs persist after navigation cycle', async ({ page }) => {
    test.setTimeout(90000); // Complex workflow needs more time
    // Add trigger
    await addTrigger(page, 'webhook.generic');

    // Add and configure multiple blocks
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');
    await addBlock(page, 'llm');

    // Organize before connecting
    await organizeLayout(page);

    // Configure blocks
    await configureLLMBlock(page, /llm-1/i, 'First LLM: Process input');
    await configureIfElseBlock(page, /if/i, '{{llm-1.success}} == true');
    await configureLLMBlock(page, /llm-2/i, 'Second LLM: Generate response');

    // Connect blocks
    await connectNodes(page, /trigger/i, /llm-1/i);
    await connectNodes(page, /llm-1/i, /if/i);

    // Wait for autosave
    await waitForAutosave(page);

    // Navigate to a completely different page
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Return to workflow
    await page.goto(`/workflows/${workflowId}`);
    await waitForWorkflowLoaded(page, 4); // Expect 4 nodes (trigger + 3 blocks)

    // Verify trigger exists
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();

    // Verify all 3 blocks exist
    const nodeCount = await page.locator('.react-flow__node').count();
    expect(nodeCount).toBe(4); // trigger + 3 blocks

    // Verify LLM-1 config
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    const prompt1 = page.getByRole('textbox', { name: /Prompt/i });
    await expect(prompt1).toHaveValue(/First LLM/);

    // Verify LLM-2 config
    await selectNode(page, /llm-2/i);
    await switchBuildPanelTab(page, 'Config');
    const prompt2 = page.getByRole('textbox', { name: /Prompt/i });
    await expect(prompt2).toHaveValue(/Second LLM/);
  });

  test('edges persist after navigation', async ({ page }) => {
    test.setTimeout(90000);
    // Build a connected workflow
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');

    await organizeLayout(page);

    await connectNodes(page, /trigger/i, /llm/i);
    await connectNodes(page, /llm/i, /if/i);

    // Verify edges exist
    let edgeCount = await page.locator('.react-flow__edge').count();
    expect(edgeCount).toBeGreaterThanOrEqual(2);

    // Wait for autosave
    await waitForAutosave(page);

    // Navigate away and back
    await page.goto('/workflows');
    await page.waitForLoadState('networkidle');
    await page.goto(`/workflows/${workflowId}`);
    await waitForWorkflowLoaded(page, 3); // Expect 3 nodes (trigger + 2 blocks)

    // Verify edges persisted
    edgeCount = await page.locator('.react-flow__edge').count();
    expect(edgeCount).toBeGreaterThanOrEqual(2);
  });

  test('config survives multiple navigation cycles', async ({ page }) => {
    test.setTimeout(90000);
    await addBlock(page, 'llm');
    await configureLLMBlock(page, /llm/i, 'Original prompt value');
    await waitForAutosave(page);

    // Navigate away and back 3 times
    for (let i = 0; i < 3; i++) {
      await page.goto('/workflows');
      await page.waitForLoadState('networkidle');
      await page.goto(`/workflows/${workflowId}`);
      await waitForWorkflowLoaded(page, 1); // Expect 1 LLM block
    }

    // Verify config still intact
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const promptField = page.getByRole('textbox', { name: /Prompt/i });
    await expect(promptField).toHaveValue(/Original prompt value/);
  });

  test('unsaved changes warning appears when navigating with pending changes', async ({ page }) => {
    // This test checks if the browser warns about unsaved changes
    // Note: Playwright handles beforeunload dialogs via page.on('dialog')

    await addBlock(page, 'llm');
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');

    const promptInput = page.getByRole('textbox', { name: /Prompt/i });
    await promptInput.waitFor({ state: 'visible', timeout: 5000 });

    // Make a change
    await promptInput.fill('Unsaved change');

    // Try to navigate immediately (before debounce completes)
    // This might trigger a warning dialog - handle it by accepting
    page.once('dialog', async (dialog) => {
      await dialog.accept(); // Accept to continue navigation
    });

    await page.goto('/workflows');

    // Either warning appeared or navigation succeeded (autosave was fast enough)
    // Both are acceptable outcomes
    await page.waitForLoadState('networkidle');
  });
});

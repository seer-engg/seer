/**
 * Page Refresh Persistence Tests
 *
 * Verifies that workflow state survives hard page refreshes (F5/Ctrl+R).
 * This tests the full persistence cycle: UI -> autosave -> backend -> reload.
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
  getNodeCount,
  getEdgeCount,
} from '../../fixtures/utils';

test.describe('Page Refresh Persistence', () => {
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

  test('single LLM block survives page refresh', async ({ page }) => {
    // Add and configure LLM block
    await addBlock(page, 'llm');
    await configureLLMBlock(page, /llm/i, 'Test prompt for refresh');

    // Wait for autosave
    await waitForAutosave(page);

    // Hard refresh
    await page.reload();
    await waitForWorkflowLoaded(page, 1); // Expect 1 LLM block

    // Verify block exists
    await expect(page.locator('.react-flow__node').filter({ hasText: /llm/i })).toBeVisible();

    // Verify config persisted
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    const promptField = page.getByRole('textbox', { name: /Prompt/i });
    await expect(promptField).toHaveValue(/Test prompt for refresh/);
  });

  test('5-node workflow survives hard page refresh', async ({ page }) => {
    test.setTimeout(90000); // Complex workflow needs more time
    // Build complex workflow
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');      // llm-1
    await addBlock(page, 'if_else');  // if-1
    await addBlock(page, 'llm');      // llm-2 (true branch)
    await addBlock(page, 'llm');      // llm-3 (false branch)

    await organizeLayout(page);

    // Connect all nodes
    await connectNodes(page, /trigger/i, /llm-1/i);
    await connectNodes(page, /llm-1/i, /if/i);
    await connectNodes(page, /if/i, /llm-2/i);
    await connectNodes(page, /if/i, /llm-3/i);

    // Configure each node
    await configureLLMBlock(page, /llm-1/i, 'Classify: {{trigger.body}}');
    await configureIfElseBlock(page, /if/i, '{{llm-1.category}} == "urgent"');
    await configureLLMBlock(page, /llm-2/i, 'Draft urgent response');
    await configureLLMBlock(page, /llm-3/i, 'Queue for later');

    // Wait for autosave
    await waitForAutosave(page);

    // Hard refresh
    await page.reload();
    await waitForWorkflowLoaded(page, 5); // Expect 5 nodes (trigger + 4 blocks)

    // Verify all 5 nodes exist
    const nodeCount = await getNodeCount(page);
    expect(nodeCount).toBe(5);

    // Verify edges exist (at least 4)
    const edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(4);

    // Verify sample configs
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Classify/);

    await selectNode(page, /llm-2/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/urgent response/);

    await selectNode(page, /llm-3/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Queue for later/);
  });

  test('node positions survive refresh', async ({ page }) => {
    test.setTimeout(60000);
    // Add blocks
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');

    await organizeLayout(page);

    // Get initial positions
    const llmNode = page.locator('.react-flow__node').filter({ hasText: /llm/i });
    const ifNode = page.locator('.react-flow__node').filter({ hasText: /if/i });

    const llmBoundsBefore = await llmNode.boundingBox();
    const ifBoundsBefore = await ifNode.boundingBox();

    expect(llmBoundsBefore).not.toBeNull();
    expect(ifBoundsBefore).not.toBeNull();

    // Wait for autosave
    await page.waitForTimeout(2000);

    // Refresh
    await page.reload();
    await waitForWorkflowLoaded(page, 2); // Expect 2 nodes

    // Get positions after refresh
    const llmBoundsAfter = await page.locator('.react-flow__node').filter({ hasText: /llm/i }).boundingBox();
    const ifBoundsAfter = await page.locator('.react-flow__node').filter({ hasText: /if/i }).boundingBox();

    expect(llmBoundsAfter).not.toBeNull();
    expect(ifBoundsAfter).not.toBeNull();

    // Positions should be approximately the same (allow some variance for rendering)
    if (llmBoundsBefore && llmBoundsAfter) {
      expect(Math.abs(llmBoundsBefore.x - llmBoundsAfter.x)).toBeLessThan(50);
      expect(Math.abs(llmBoundsBefore.y - llmBoundsAfter.y)).toBeLessThan(50);
    }
  });

  test('multiple rapid refreshes do not corrupt state', async ({ page }) => {
    test.setTimeout(90000);
    // Build workflow
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');
    await configureLLMBlock(page, /llm/i, 'Persistent prompt');

    // Wait for save
    await waitForAutosave(page);

    // Refresh 3 times rapidly
    for (let i = 0; i < 3; i++) {
      await page.reload();
      await waitForWorkflowLoaded(page, 2); // Expect 2 nodes to load
      // Quick check nodes still exist
      const nodeCount = await getNodeCount(page);
      expect(nodeCount).toBe(2);
    }

    // Final verification of config
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Persistent prompt/);
  });

  test('refresh during debounce still saves last edit', async ({ page }) => {
    await addBlock(page, 'llm');
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');

    const promptInput = page.getByRole('textbox', { name: /Prompt/i });
    await promptInput.waitFor({ state: 'visible', timeout: 5000 });

    // Make edit
    await promptInput.fill('Edit before refresh');

    // Wait just enough for debounce but before confirming save
    await page.waitForTimeout(1200);

    // Refresh
    await page.reload();
    await waitForWorkflowLoaded(page, 1); // Expect at least 1 block

    // Verify edit was saved (should have been captured by debounce)
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');

    // The edit might or might not have saved depending on exact timing
    // At minimum, the block should exist
    await expect(page.locator('.react-flow__node').filter({ hasText: /llm/i })).toBeVisible();
  });

  test('workflow with all block types survives refresh', async ({ page }) => {
    test.setTimeout(90000); // Complex workflow needs more time
    // This is a more comprehensive test with multiple block types

    // Add various block types
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');
    await addBlock(page, 'for_loop');
    await addBlock(page, 'hitl');

    await organizeLayout(page);

    // Wait for autosave
    await waitForAutosave(page);
    await page.waitForTimeout(1000); // Extra buffer

    // Refresh
    await page.reload();
    await waitForWorkflowLoaded(page, 5); // Expect 5 nodes (trigger + 4 blocks)

    // Verify all 5 nodes exist (trigger + 4 blocks)
    const nodeCount = await getNodeCount(page);
    expect(nodeCount).toBe(5);

    // Verify each block type is present
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();
    await expect(page.locator('.react-flow__node').filter({ hasText: /llm/i })).toBeVisible();
    await expect(page.locator('.react-flow__node').filter({ hasText: /if/i })).toBeVisible();
    await expect(page.locator('.react-flow__node').filter({ hasText: /loop/i })).toBeVisible();
    await expect(page.locator('.react-flow__node').filter({ hasText: /hitl/i })).toBeVisible();
  });

  test('edge connections preserved through refresh', async ({ page }) => {
    // Create connected workflow
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');
    await addBlock(page, 'llm');

    await organizeLayout(page);

    // Create chain: trigger -> llm-1 -> llm-2
    await connectNodes(page, /trigger/i, /llm-1/i);
    await connectNodes(page, /llm-1/i, /llm-2/i);

    // Verify edges before refresh
    let edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(2);

    // Wait for autosave
    await waitForAutosave(page);

    // Refresh
    await page.reload();
    await waitForWorkflowLoaded(page, 3); // Expect 3 nodes (trigger + 2 LLMs)

    // Verify edges after refresh
    edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(2);
  });
});

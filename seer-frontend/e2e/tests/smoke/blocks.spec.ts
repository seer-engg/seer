/**
 * Block Rendering Smoke Tests
 *
 * Fast validation that all block types can be added and render correctly.
 * These tests should complete in under 30 seconds total.
 */
import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addBlock,
  addTrigger,
} from '../../fixtures/utils';

// All block types available in the workflow builder
const BLOCK_TYPES = [
  'llm',
  'if_else',
  'for_loop',
  'hitl',
  'image_gen',
  'browser',
  'mcp',
] as const;

// Trigger types available for smoke testing (OAuth-free)
// Note: Gmail, Slack, etc. require OAuth and are tested in integration tests

test.describe('Block Rendering Smoke Tests', () => {
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

  // Generate a test for each block type
  for (const blockType of BLOCK_TYPES) {
    test(`${blockType} block adds and renders`, async ({ page }) => {
      // Add the block
      await addBlock(page, blockType);

      // Verify exactly one node was added
      await expect(page.locator('.react-flow__node')).toHaveCount(1);

      // Verify the block has expected content/label
      const node = page.locator('.react-flow__node').first();
      await expect(node).toBeVisible();

      // Block should be selectable - clicking opens Config panel
      await node.click();
      // Verify Config tab is selected (indicates node was clicked)
      await expect(page.getByRole('tab', { name: 'Config', selected: true })).toBeVisible();
    });
  }

  test('webhook trigger adds and renders', async ({ page }) => {
    await addTrigger(page, 'webhook.generic');

    await expect(page.locator('.react-flow__node')).toHaveCount(1);
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();
  });

  test('multiple different blocks can coexist', async ({ page }) => {
    // Add several different block types
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');
    await addBlock(page, 'for_loop');

    // All three should be visible
    await expect(page.locator('.react-flow__node')).toHaveCount(3);
    await expect(page.locator('.react-flow__node').filter({ hasText: /llm/i })).toBeVisible();
    await expect(page.locator('.react-flow__node').filter({ hasText: /if/i })).toBeVisible();
    await expect(page.locator('.react-flow__node').filter({ hasText: /loop/i })).toBeVisible();
  });

  test('trigger plus blocks can coexist', async ({ page }) => {
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');
    await addBlock(page, 'if_else');

    await expect(page.locator('.react-flow__node')).toHaveCount(3);
  });

  test('blocks render with correct visual elements', async ({ page }) => {
    // Add LLM block and verify it has connection handles
    await addBlock(page, 'llm');

    const llmNode = page.locator('.react-flow__node').first();

    // Should have connection handles (input and output)
    const handles = llmNode.locator('.react-flow__handle');
    const handleCount = await handles.count();
    expect(handleCount).toBeGreaterThanOrEqual(2); // At least input + output
  });

  test('if_else block shows branch labels', async ({ page }) => {
    await addBlock(page, 'if_else');

    const ifNode = page.locator('.react-flow__node').filter({ hasText: /if/i });

    // Should show True and False branch labels
    await expect(ifNode.getByText('True')).toBeVisible();
    await expect(ifNode.getByText('False')).toBeVisible();
  });

  test('for_loop block renders correctly', async ({ page }) => {
    await addBlock(page, 'for_loop');

    const forNode = page.locator('.react-flow__node').filter({ hasText: /loop/i });
    await expect(forNode).toBeVisible();

    // For loop nodes have loop-specific handles
    const handles = forNode.locator('.react-flow__handle');
    const handleCount = await handles.count();
    expect(handleCount).toBeGreaterThanOrEqual(2);
  });

  test('hitl block renders approval interface indicator', async ({ page }) => {
    await addBlock(page, 'hitl');

    const hitlNode = page.locator('.react-flow__node').filter({ hasText: /hitl/i });
    await expect(hitlNode).toBeVisible();
  });

  test('blocks can be deleted', async ({ page }) => {
    await addBlock(page, 'llm');
    await expect(page.locator('.react-flow__node')).toHaveCount(1);

    // Select and delete
    const node = page.locator('.react-flow__node').first();
    await node.click();
    await page.keyboard.press('Delete');

    // May need to also try Backspace
    await page.keyboard.press('Backspace');

    await page.waitForTimeout(500);

    // Node should be gone
    await expect(page.locator('.react-flow__node')).toHaveCount(0);
  });
});

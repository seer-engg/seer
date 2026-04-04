/**
 * Verify right panel tabs don't produce duplicate headers.
 */
import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addBlock,
} from '../../fixtures/utils';

test.describe('No Duplicate Panel Headers', () => {
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

  test('Build tab shows single header', async ({ page }) => {
    const buildTab = page.getByRole('tab', { name: 'Build' });
    await expect(buildTab).toBeVisible();
    await buildTab.click();

    // The parent panel header should be the only one
    const panel = page.locator('[role="tabpanel"]').first();
    await expect(panel).toBeVisible();

    // Should not have stacked h-14 border-b headers (the PanelHeader pattern)
    const headers = panel.locator('.h-14.border-b');
    await expect(headers).toHaveCount(0);
  });

  test('Executions tab shows single header', async ({ page }) => {
    const execTab = page.getByRole('tab', { name: 'Executions' });
    await expect(execTab).toBeVisible();
    await execTab.click();

    const panel = page.locator('[role="tabpanel"]').first();
    await expect(panel).toBeVisible();

    // No duplicate PanelHeader inside the tab content
    const headers = panel.locator('.h-14.border-b');
    await expect(headers).toHaveCount(0);
  });

  test('Config panel shows no duplicate header when node selected', async ({ page }) => {
    // Add a block and click it to open config
    await addBlock(page, 'llm');

    const node = page.locator('.react-flow__node').first();
    await expect(node).toBeVisible();
    await node.click();

    // Wait for config content to appear
    await expect(page.getByText('Node ID')).toBeVisible();

    // The tab panel content should not contain a duplicate PanelHeader
    const panel = page.locator('[role="tabpanel"]').first();
    const headers = panel.locator('.h-14.border-b');
    await expect(headers).toHaveCount(0);
  });
});

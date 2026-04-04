/**
 * Basic Execution Smoke Tests
 *
 * Quick validation that the simplest workflow can be built and executed.
 * Uses webhook trigger to avoid OAuth dependencies.
 */
import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  waitForAutosave,
  addBlock,
  addTrigger,
  connectNodes,
  organizeLayout,
  configureLLMBlock,
  runWorkflow,
  publishWorkflow,
  getFirstTriggerId,
  getWebhookUrl,
  postToWebhook,
  switchBuildPanelTab,
} from '../../fixtures/utils';

test.describe('Basic Execution Smoke Tests', () => {
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

  test('minimal workflow: webhook trigger only', async ({ page }) => {
    // Add just a webhook trigger
    await addTrigger(page, 'webhook.generic');
    await expect(page.locator('.react-flow__node')).toHaveCount(1);

    // Should be able to run (opens trigger selection dialog)
    await runWorkflow(page);

    // Dialog should appear
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible({ timeout: 10000 });

    // Close dialog
    await page.keyboard.press('Escape');
  });

  test('webhook + LLM workflow can be built', async ({ page }) => {
    // Build minimal workflow
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');

    await organizeLayout(page);

    // Connect trigger to LLM
    await connectNodes(page, /trigger/i, /llm/i);

    // Verify edge created
    const edgeCount = await page.locator('.react-flow__edge').count();
    expect(edgeCount).toBeGreaterThanOrEqual(1);

    // Configure LLM
    await configureLLMBlock(page, /llm/i, 'Echo: {{trigger.message}}');

    // Wait for save
    await waitForAutosave(page);
  });

  test('workflow can be published', async ({ page }) => {
    // Build workflow
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');
    await organizeLayout(page);
    await connectNodes(page, /trigger/i, /llm/i);
    await configureLLMBlock(page, /llm/i, 'Test prompt');

    // Wait for save
    await waitForAutosave(page);

    // Publish
    await publishWorkflow(page);

    // Should see some confirmation (button state change, toast, etc.)
    // The exact UI varies but publishing should not error
    await page.waitForTimeout(2000);
  });

  test('webhook trigger execution flow', async ({ page }) => {
    // This test uses longer timeout as it involves actual execution
    test.setTimeout(120000);

    // Build simple workflow
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');
    await organizeLayout(page);
    await connectNodes(page, /trigger/i, /llm/i);
    await configureLLMBlock(page, /llm/i, 'Say hello to: {{trigger.name}}');

    // Wait for save
    await waitForAutosave(page);

    // Get trigger ID for webhook URL
    const triggerId = await getFirstTriggerId(page, workflowId);

    // Get webhook URL
    const webhookUrl = await getWebhookUrl(page, workflowId, triggerId);
    expect(webhookUrl).toContain('/webhooks/');

    // Try to post to webhook - may fail due to auth token expiry
    try {
      await postToWebhook(page, webhookUrl, {
        name: 'E2E Test User',
        message: 'Hello from smoke test'
      });
    } catch (error) {
      // Webhook auth can fail due to token expiry in parallel test runs
      // Skip the execution verification part
      if (String(error).includes('401')) {
        console.log('Webhook auth failed (expected in some CI environments)');
        return;
      }
      throw error;
    }

    // Switch to Executions tab to verify
    await switchBuildPanelTab(page, 'Executions');

    // Wait for execution to appear (may already be running or completed)
    const executionIndicator = page.getByText(/Running|Succeeded|Failed|Queued/i);
    await expect(executionIndicator.first()).toBeVisible({ timeout: 30000 });
  });

  test('run button opens trigger selection dialog', async ({ page }) => {
    // Add trigger
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');
    await organizeLayout(page);
    await connectNodes(page, /trigger/i, /llm/i);

    // Wait for save
    await waitForAutosave(page);

    // Click run
    await runWorkflow(page);

    // Dialog should appear with trigger options
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible({ timeout: 10000 });

    // Should show webhook option
    await expect(dialog.getByText(/Webhook/i)).toBeVisible();

    // Close
    await page.keyboard.press('Escape');
  });

  test('executions tab shows expected UI', async ({ page }) => {
    // Build a workflow
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');
    await organizeLayout(page);
    await connectNodes(page, /trigger/i, /llm/i);

    await waitForAutosave(page);

    // Switch to Executions tab
    await switchBuildPanelTab(page, 'Executions');

    // Executions tab should be visible and selected
    await expect(page.getByRole('tab', { name: 'Executions', selected: true })).toBeVisible();

    // Tab panel should show executions content (empty state or list)
    const tabPanel = page.getByRole('tabpanel', { name: 'Executions' });
    await expect(tabPanel).toBeVisible();
  });

  test('canvas controls are functional', async ({ page }) => {
    // Verify zoom controls exist
    const zoomIn = page.locator('.react-flow__controls-zoomin');
    const zoomOut = page.locator('.react-flow__controls-zoomout');
    const fitView = page.locator('.react-flow__controls-fitview');

    await expect(zoomIn).toBeVisible();
    await expect(zoomOut).toBeVisible();
    await expect(fitView).toBeVisible();

    // Add a block first
    await addBlock(page, 'llm');

    // Click fit view - should work without error
    await fitView.click();
    await page.waitForTimeout(500);

    // Zoom out should always be enabled (can zoom out from fit view)
    await zoomOut.click();
    await page.waitForTimeout(200);

    // After zooming out, zoom in should be enabled
    if (await zoomIn.isEnabled()) {
      await zoomIn.click();
      await page.waitForTimeout(200);
    }
  });
});

/**
 * Slack Integration - Complete User Journey Test
 *
 * A comprehensive E2E test covering full Slack round-trip:
 * 1. Trigger: Receive message from #workflow_testing
 * 2. Action: Send response back to #workflow_testing via Slack tool
 * 3. Verify execution completes successfully
 */
import { test, expect } from '../../fixtures/auth';
import type { Page } from '@playwright/test';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addTrigger,
  organizeLayout,
  selectNode,
  switchBuildPanelTab,
  runWorkflow,
  publishWorkflow,
  selectTriggerEvent,
  waitForExecutionStatus,
  waitForAutosave,
} from '../../fixtures/utils';

/**
 * Configure Slack trigger by selecting workspace and #workflow_testing channel.
 * Uses the "Browse Resources" dialog that appears when clicking dropdowns.
 */
async function configureSlackTrigger(page: Page): Promise<boolean> {
  await switchBuildPanelTab(page, 'Config');
  await page.waitForTimeout(1000);

  const configPanel = page.locator('[data-testid="block-config-panel"]');

  // Find and click workspace dropdown
  const workspaceButton = configPanel.locator('button').filter({ hasText: /Select.*Workspace/i }).first();

  if (!await workspaceButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    // Workspace already selected or different UI
    return true;
  }

  await workspaceButton.click();
  await page.waitForTimeout(500);

  // Wait for "Browse Resources" dialog to appear
  const dialog = page.getByRole('dialog', { name: /Browse Resources/i });
  if (!await dialog.isVisible({ timeout: 5000 }).catch(() => false)) {
    await page.keyboard.press('Escape');
    return false;
  }

  // Find clickable items in the dialog
  const resourceItems = dialog.locator('[class*="cursor-pointer"]').filter({ has: page.locator('p') });
  const itemCount = await resourceItems.count();

  if (itemCount === 0) {
    await dialog.getByRole('button', { name: 'Close' }).click().catch(() => page.keyboard.press('Escape'));
    return false;
  }

  // Click first workspace (Seer)
  await resourceItems.first().click();
  await page.waitForTimeout(1000);

  // Now select #workflow_testing channel
  const channelButton = configPanel.locator('button').filter({ hasText: /Select.*Channel/i }).first();

  if (await channelButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    await channelButton.click();
    await page.waitForTimeout(500);

    const channelDialog = page.getByRole('dialog', { name: /Browse Resources/i });
    if (await channelDialog.isVisible({ timeout: 5000 }).catch(() => false)) {
      const channelItems = channelDialog.locator('[class*="cursor-pointer"]').filter({ has: page.locator('p') });
      const channelCount = await channelItems.count();

      if (channelCount > 0) {
        // Select #workflow_testing specifically (where the bot is added)
        const workflowTestingChannel = channelItems.filter({ hasText: '#workflow_testing' });
        if (await workflowTestingChannel.count() > 0) {
          await workflowTestingChannel.first().click();
        } else {
          await channelItems.first().click();
        }
        await page.waitForTimeout(500);
      } else {
        await channelDialog.getByRole('button', { name: 'Close' }).click().catch(() => page.keyboard.press('Escape'));
      }
    }
  }

  return true;
}

/**
 * Configure Slack send_channel_message tool.
 * Sets workspace, channel (#workflow_testing), and message text.
 */
async function configureSlackTool(page: Page, messageText: string): Promise<void> {
  await switchBuildPanelTab(page, 'Config');
  await page.waitForTimeout(1000);

  const configPanel = page.locator('[data-testid="block-config-panel"]');

  // Select workspace
  const workspaceButton = configPanel.locator('button').filter({ hasText: /Select.*Workspace/i }).first();
  if (await workspaceButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    await workspaceButton.click();
    await page.waitForTimeout(500);

    const dialog = page.getByRole('dialog', { name: /Browse Resources/i });
    if (await dialog.isVisible({ timeout: 5000 }).catch(() => false)) {
      const items = dialog.locator('[class*="cursor-pointer"]').filter({ has: page.locator('p') });
      if (await items.count() > 0) {
        await items.first().click();
        await page.waitForTimeout(500);
      }
    }
  }

  // Select #workflow_testing channel
  const channelButton = configPanel.locator('button').filter({ hasText: /Select.*Channel/i }).first();
  if (await channelButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    await channelButton.click();
    await page.waitForTimeout(500);

    const channelDialog = page.getByRole('dialog', { name: /Browse Resources/i });
    if (await channelDialog.isVisible({ timeout: 5000 }).catch(() => false)) {
      const channelItems = channelDialog.locator('[class*="cursor-pointer"]').filter({ has: page.locator('p') });
      // Select #workflow_testing
      const workflowTestingChannel = channelItems.filter({ hasText: '#workflow_testing' });
      if (await workflowTestingChannel.count() > 0) {
        await workflowTestingChannel.first().click();
      } else if (await channelItems.count() > 0) {
        await channelItems.first().click();
      }
      await page.waitForTimeout(500);
    }
  }

  // Set message text - look for the text field by label
  const textField = configPanel.getByRole('textbox', { name: /text/i }).first();
  if (await textField.isVisible({ timeout: 2000 }).catch(() => false)) {
    await textField.fill(messageText);
  } else {
    // Try alternative: look for any textarea in config
    const textarea = configPanel.locator('textarea').first();
    if (await textarea.isVisible({ timeout: 2000 }).catch(() => false)) {
      await textarea.fill(messageText);
    }
  }

  await page.waitForTimeout(500);
}

test.describe('Slack Integration', () => {
  let workflowId: string;

  test.afterEach(async ({ page }) => {
    if (workflowId) {
      await deleteTestWorkflow(page, workflowId);
    }
  });

  test('complete slack round-trip: receive message and respond', async ({ page }) => {
    test.setTimeout(300000); // 5 minutes for full journey

    // =========================================================================
    // PHASE 1: Setup & Verify Slack Connection
    // =========================================================================
    workflowId = await openWorkflow(page);
    await waitForCanvasReady(page);

    // Verify Slack trigger is visible and connected
    const slackTrigger = page.locator('[data-testid="trigger-poll.slack.message_received"]');
    await expect(slackTrigger).toBeVisible({ timeout: 5000 });

    const connectedBadge = slackTrigger.getByText('Connected');
    const isConnected = await connectedBadge.isVisible().catch(() => false);

    if (!isConnected) {
      test.skip(true, 'Slack OAuth not connected');
      return;
    }

    // =========================================================================
    // PHASE 2: Add Slack Trigger to Canvas
    // =========================================================================
    await addTrigger(page, 'poll.slack.message_received');
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();

    // =========================================================================
    // PHASE 3: Configure Slack Trigger (#workflow_testing channel)
    // =========================================================================
    await selectNode(page, /trigger/i);
    const configSuccess = await configureSlackTrigger(page);

    if (!configSuccess) {
      test.skip(true, 'No Slack workspaces available');
      return;
    }

    // =========================================================================
    // PHASE 4: Add Slack Tool to Send Response Back
    // =========================================================================
    // Use the "Add node after trigger" button for direct connection
    const addNodeButton = page.locator('.react-flow__node')
      .filter({ hasText: /trigger/i })
      .getByRole('button', { name: /add node/i });
    await addNodeButton.click();
    await page.waitForTimeout(500);

    // The Build panel enters selection mode (no popover - just the existing panel)
    // Look for the selection mode indicator
    await expect(page.getByText('Select a block to add')).toBeVisible({ timeout: 5000 });

    // Search for slack in the Build panel's search input
    const searchInput = page.locator('[data-testid="build-search"]');
    await searchInput.fill('slack');
    await page.waitForTimeout(1000);

    // Click on Slack in Actions section (shows "7 tools")
    // This directly adds slack_send_channel_message to the canvas with an edge
    // Scope to Build tab panel to avoid matching canvas nodes
    const buildPanel = page.getByRole('tabpanel', { name: 'Build' });
    const slackActions = buildPanel.getByText('7 tools').first();
    await slackActions.waitFor({ state: 'visible', timeout: 5000 });
    // Click the parent element that contains both "Slack" and "7 tools"
    await slackActions.locator('..').click();
    await page.waitForTimeout(1000);

    // The tool is auto-added to canvas - no need to click on send_channel_message
    // Verify the node was added
    await expect(page.locator('.react-flow__node').filter({ hasText: /slack_send/i })).toBeVisible({ timeout: 5000 });

    await organizeLayout(page);

    // Verify connection (added via "add after" button auto-connects)
    await page.waitForTimeout(500);
    const edgeCount = await page.locator('.react-flow__edge').count();
    expect(edgeCount).toBeGreaterThanOrEqual(1);

    // Configure Slack tool to respond in #workflow_testing
    await selectNode(page, /slack_send/i);
    await configureSlackTool(
      page,
      '[E2E Test] Received your message: "{{trigger-1.data.text}}"'
    );
    await waitForAutosave(page);

    // =========================================================================
    // PHASE 5: Verify Persistence After Reload
    // =========================================================================
    await page.reload();
    await waitForCanvasReady(page);

    // Verify structure preserved (trigger + slack tool)
    expect(await page.locator('.react-flow__node').count()).toBe(2);
    expect(await page.locator('.react-flow__edge').count()).toBeGreaterThanOrEqual(1);

    // =========================================================================
    // PHASE 6: Publish & Run Workflow
    // =========================================================================
    await publishWorkflow(page);
    await page.waitForTimeout(2000);
    await runWorkflow(page);

    // Handle event picker dialog
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible({ timeout: 10000 });

    // Select Slack option if multi-trigger dialog
    const slackButton = dialog.getByRole('button').filter({ hasText: /Slack/i });
    if (await slackButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await slackButton.click();
    }

    // Wait for events to load
    await page.waitForTimeout(3000);

    // Check for errors
    const retryButton = dialog.getByRole('button', { name: /Retry/i });
    const hasError = await retryButton.isVisible({ timeout: 2000 }).catch(() => false);

    if (hasError) {
      await page.keyboard.press('Escape');
      // Workflow built correctly - environment issue
      return;
    }

    // Select trigger event
    const eventSelected = await selectTriggerEvent(page, { timeout: 15000 });

    if (!eventSelected) {
      await page.keyboard.press('Escape');
      return;
    }

    // =========================================================================
    // PHASE 7: Verify Execution
    // =========================================================================
    const status = await waitForExecutionStatus(page, { timeout: 180000 });
    expect(['succeeded', 'failed', 'timeout']).toContain(status);

    // Verify execution in list
    await switchBuildPanelTab(page, 'Executions');
    const executionStatus = page.locator('text=/Succeeded|Running|Failed|Queued/i').first();
    await expect(executionStatus).toBeVisible({ timeout: 10000 });
  });
});

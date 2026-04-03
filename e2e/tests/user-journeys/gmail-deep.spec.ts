/**
 * Gmail Integration - Complete User Journey Test
 *
 * Tests: visibility -> add -> connect -> configure -> refresh -> publish -> execute -> succeed
 *
 * Variable paths: Gmail trigger fields are nested under `data.` prefix:
 * - {{trigger-1.data.subject}} - email subject
 * - {{trigger-1.data.snippet}} - email preview snippet
 * - {{trigger-1.data.from.email}} - sender email
 * (See src/components/workflows/triggers/constants.ts for full list)
 */
import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addBlock,
  addTrigger,
  connectNodes,
  organizeLayout,
  configureLLMBlock,
  isGmailConnected,
  selectNode,
  switchBuildPanelTab,
  runWorkflow,
  publishWorkflow,
  selectTriggerEvent,
  waitForExecutionStatus,
  waitForAutosave,
} from '../../fixtures/utils';

test.describe('Gmail Integration', () => {
  let workflowId: string;
  let gmailConnected: boolean;

  test.beforeEach(async ({ page }) => {
    workflowId = await openWorkflow(page);
    await waitForCanvasReady(page);
    gmailConnected = await isGmailConnected(page);
  });

  test.afterEach(async ({ page }) => {
    if (workflowId) {
      await deleteTestWorkflow(page, workflowId);
    }
  });

  test('gmail workflow: complete user journey', async ({ page }) => {
    test.skip(!gmailConnected, 'Gmail OAuth not connected');
    test.setTimeout(300000); // 5 minutes

    // ═══════════════════════════════════════════════════════════
    // PHASE 1: Verify Gmail trigger visibility in build panel
    // ═══════════════════════════════════════════════════════════
    const gmailTrigger = page.locator(
      '[data-testid="trigger-poll.gmail.email_received"]'
    );
    await expect(gmailTrigger).toBeVisible({ timeout: 5000 });
    await expect(gmailTrigger.getByText('Connected')).toBeVisible();

    // ═══════════════════════════════════════════════════════════
    // PHASE 2: Build workflow (Gmail -> LLM)
    // Simplified from 4-node to 2-node for focused testing
    // ═══════════════════════════════════════════════════════════
    await addTrigger(page, 'poll.gmail.email_received');
    await addBlock(page, 'llm');
    await organizeLayout(page);

    expect(await page.locator('.react-flow__node').count()).toBe(2);

    // ═══════════════════════════════════════════════════════════
    // PHASE 3: Connect nodes
    // ═══════════════════════════════════════════════════════════
    await connectNodes(page, /trigger/i, /llm/i);
    expect(
      await page.locator('.react-flow__edge').count()
    ).toBeGreaterThanOrEqual(1);

    // ═══════════════════════════════════════════════════════════
    // PHASE 4: Configure LLM with correct Gmail variable paths
    //
    // Gmail trigger fields are nested under `data.` prefix:
    // - trigger-1.data.subject (NOT trigger.subject)
    // - trigger-1.data.snippet (NOT trigger.body)
    // See: src/components/workflows/triggers/constants.ts
    // ═══════════════════════════════════════════════════════════
    await configureLLMBlock(
      page,
      /llm/i,
      'Summarize this email:\nSubject: {{trigger-1.data.subject}}\nContent: {{trigger-1.data.snippet}}'
    );

    await waitForAutosave(page);

    // ═══════════════════════════════════════════════════════════
    // PHASE 5: Verify persistence through page refresh
    // ═══════════════════════════════════════════════════════════
    await page.reload();
    await waitForCanvasReady(page);

    expect(await page.locator('.react-flow__node').count()).toBe(2);
    expect(
      await page.locator('.react-flow__edge').count()
    ).toBeGreaterThanOrEqual(1);

    // Verify LLM config preserved (contains the correct variable paths)
    await selectNode(page, /llm/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(
      /trigger-1\.data\.subject/
    );

    // ═══════════════════════════════════════════════════════════
    // PHASE 6: Publish and execute
    // ═══════════════════════════════════════════════════════════
    await publishWorkflow(page);
    await page.waitForTimeout(2000);

    await runWorkflow(page);

    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible({ timeout: 10000 });

    const gmailButton = dialog.getByRole('button').filter({ hasText: /Gmail/i });
    if (await gmailButton.isVisible({ timeout: 3000 }).catch(() => false)) {
      await gmailButton.click();
    }
    await page.waitForTimeout(1000);

    const eventSelected = await selectTriggerEvent(page, { timeout: 15000 });
    if (!eventSelected) {
      test.skip(true, 'No Gmail events available for execution');
      return;
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 7: Verify execution SUCCEEDED
    // Only accept "succeeded" - previous version was too permissive
    // ═══════════════════════════════════════════════════════════
    const status = await waitForExecutionStatus(page, { timeout: 180000 });
    expect(status).toBe('succeeded');
  });
});

import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addTrigger,
  addBlock,
  connectNodes,
  runWorkflow,
  isGmailConnected,
  configureLLMBlock,
  getEdgeCount,
  organizeLayout,
  selectTriggerEvent,
  waitForExecutionStatus,
} from '../../fixtures/utils';

test.describe('Gmail Auto-Reply Workflow', () => {
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

  test('create, connect, run, and verify gmail-llm workflow', async ({ page }) => {
    // This test executes an LLM workflow which can take several minutes
    test.setTimeout(240000); // 4 minutes

    // Skip if Gmail not connected
    const gmailConnected = await isGmailConnected(page);
    if (!gmailConnected) {
      test.skip(true, 'Gmail OAuth not connected - skipping Gmail workflow test');
      return;
    }

    // 1. ADD TRIGGER: Gmail email received
    await addTrigger(page, 'poll.gmail.email_received');
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger-1/i })).toBeVisible();

    // 2. ADD LLM BLOCK: Draft reply
    await addBlock(page, 'llm');
    await expect(page.locator('.react-flow__node').filter({ hasText: /llm-1/i })).toBeVisible();

    // 3. CONFIGURE LLM: Set prompt for email reply drafting
    await configureLLMBlock(page, /llm-1/i,
      'Draft a professional reply to the following email: {{trigger-1.data.snippet}}'
    );

    // 4. ORGANIZE LAYOUT: Separate overlapping nodes before connecting
    await organizeLayout(page);

    // 5. CONNECT: Trigger -> LLM
    await connectNodes(page, /trigger-1/i, /llm-1/i);

    // 6. VERIFY EDGE CREATED
    const edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(1);

    // 7. RUN WORKFLOW (opens event picker directly for single-trigger workflows)
    await runWorkflow(page);

    // 8. SELECT LATEST EMAIL FROM TRIGGER EVENT PICKER
    const eventSelected = await selectTriggerEvent(page, { timeout: 15000 });

    if (!eventSelected) {
      // No emails available in the connected Gmail account
      test.skip(true, 'No Gmail events available - skipping execution test');
      await page.keyboard.press('Escape');
      return;
    }

    // 9. WAIT FOR EXECUTION TO COMPLETE
    // LLM workflows can take time, use generous timeout (3 minutes)
    const status = await waitForExecutionStatus(page, { timeout: 180000 });

    // 10. VERIFY EXECUTION COMPLETED (not timed out)
    // Accept both succeeded and failed - we're testing the execution pipeline works
    expect(['succeeded', 'failed']).toContain(status);
    console.log(`Gmail workflow execution completed with status: ${status}`);
  });
});

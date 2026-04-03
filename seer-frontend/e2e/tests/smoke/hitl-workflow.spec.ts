import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addTrigger,
  addBlock,
  connectNodes,
  organizeLayout,
  getNodeCount,
  getEdgeCount,
  configureHITLBlock,
  configureLLMBlock,
  runWorkflow,
  selectNode,
  switchBuildPanelTab,
} from '../../fixtures/utils';

test.describe('HITL Workflow', () => {
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

  test('create workflow with HITL block and verify structure', async ({ page }) => {
    // 1. ADD TRIGGER: Webhook (no OAuth required)
    await addTrigger(page, 'webhook.generic');
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();

    // 2. ADD HITL BLOCK (Human-in-the-Loop for approval)
    await addBlock(page, 'hitl');
    const hitlNode = page.locator('.react-flow__node').filter({ hasText: /hitl/i });
    await expect(hitlNode).toBeVisible();

    // 3. ADD LLM BLOCK (runs after HITL approval)
    await addBlock(page, 'llm');
    await expect(page.locator('.react-flow__node').filter({ hasText: /llm/i })).toBeVisible();

    // Verify we have 3 nodes: trigger + hitl + llm
    const nodeCount = await getNodeCount(page);
    expect(nodeCount).toBe(3);

    // 4. ORGANIZE LAYOUT before configuring and connecting
    await organizeLayout(page);

    // 5. CONFIGURE HITL: Set approval title and description
    await configureHITLBlock(page, /hitl/i, {
      title: 'Test Approval',
      description: 'Please approve this test workflow execution'
    });

    // 6. CONFIGURE LLM: Simple prompt to verify workflow continues after approval
    await configureLLMBlock(
      page,
      /llm/i,
      'The approval was granted. Respond with "Workflow completed after approval".'
    );

    // 7. CONNECT: Trigger -> HITL
    await connectNodes(page, /trigger/i, /hitl/i);

    // 8. CONNECT: HITL -> LLM
    await connectNodes(page, /hitl/i, /llm/i);

    // 9. VERIFY EDGES CREATED
    const edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(2);

    // 10. RUN WORKFLOW (opens trigger event selection dialog)
    await runWorkflow(page);

    // 11. VERIFY TRIGGER EVENT DIALOG APPEARS
    const triggerDialog = page.getByRole('dialog', { name: /Select Trigger Event/i });
    await expect(triggerDialog).toBeVisible({ timeout: 10000 });

    // 12. VERIFY WEBHOOK OPTION IS SHOWN (confirms trigger and structure)
    const webhookOption = triggerDialog.getByText('Webhook');
    await expect(webhookOption).toBeVisible();

    // 13. CLOSE DIALOG (workflow structure validated)
    const closeButton = triggerDialog.getByRole('button', { name: /Close/i });
    await closeButton.click();

    console.log('HITL workflow structure validated successfully');
  });

  test('HITL node shows correct configuration fields', async ({ page }) => {
    // Add HITL block
    await addBlock(page, 'hitl');

    const hitlNode = page.locator('.react-flow__node').filter({ hasText: /hitl/i });
    await expect(hitlNode).toBeVisible();

    // Select the node and check config panel
    await selectNode(page, /hitl/i);
    await switchBuildPanelTab(page, 'Config');

    // Verify config fields are visible (use textbox role, label is "Title*" for required)
    const titleField = page.getByRole('textbox', { name: /^Title\*?$/i });
    await expect(titleField).toBeVisible({ timeout: 5000 });

    const descriptionField = page.getByRole('textbox', { name: /^Description$/i });
    await expect(descriptionField).toBeVisible();

    // Verify timeout field exists (it's a number input)
    const timeoutField = page.getByRole('spinbutton', { name: /Timeout/i });
    await expect(timeoutField).toBeVisible();
  });

  test('workflow with HITL between trigger and LLM', async ({ page }) => {
    // 1. ADD TRIGGER: Webhook
    await addTrigger(page, 'webhook.generic');

    // 2. ADD HITL BLOCK
    await addBlock(page, 'hitl');

    // 3. ADD LLM BLOCK
    await addBlock(page, 'llm');

    // 4. ORGANIZE and CONNECT
    await organizeLayout(page);
    await connectNodes(page, /trigger/i, /hitl/i);
    await connectNodes(page, /hitl/i, /llm/i);

    // 5. CONFIGURE HITL
    await configureHITLBlock(page, /hitl/i, {
      title: 'Quick Approval Test',
      description: 'Approve to continue'
    });

    // 6. VERIFY STRUCTURE
    const nodeCount = await getNodeCount(page);
    expect(nodeCount).toBe(3);

    const edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(2);

    // 7. RUN WORKFLOW
    await runWorkflow(page);

    // 8. VERIFY TRIGGER EVENT DIALOG
    const triggerDialog = page.getByRole('dialog', { name: /Select Trigger Event/i });
    await expect(triggerDialog).toBeVisible({ timeout: 10000 });

    // 9. CLOSE DIALOG
    const closeButton = triggerDialog.getByRole('button', { name: /Close/i });
    await closeButton.click();

    console.log('HITL between trigger and LLM workflow validated successfully');
  });
});

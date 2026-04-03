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
  configureForLoopBlock,
  configureLLMBlock,
  runWorkflow,
} from '../../fixtures/utils';

test.describe('For Loop Workflow', () => {
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

  test('create workflow with for loop and verify structure', async ({ page }) => {
    // 1. ADD TRIGGER: Webhook (no OAuth needed)
    await addTrigger(page, 'webhook.generic');
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();

    // 2. ADD FOR LOOP BLOCK
    await addBlock(page, 'for_loop');
    const forLoopNode = page.locator('.react-flow__node').filter({ hasText: /loop/i });
    await expect(forLoopNode).toBeVisible();

    // 3. VERIFY FOR LOOP HANDLES EXIST (Loop and Exit)
    await expect(forLoopNode.getByText('Loop', { exact: true })).toBeVisible();
    await expect(forLoopNode.getByText('Exit', { exact: true })).toBeVisible();

    // 4. ADD LLM BLOCK (inside the loop - will run for each item)
    await addBlock(page, 'llm');
    await expect(page.locator('.react-flow__node').filter({ hasText: /llm/i })).toBeVisible();

    // Verify we have 3 nodes: trigger + for_loop + llm
    const nodeCount = await getNodeCount(page);
    expect(nodeCount).toBe(3);

    // 5. ORGANIZE LAYOUT before configuring and connecting
    await organizeLayout(page);

    // 6. CONFIGURE FOR LOOP: Set array variable to reference webhook payload
    await configureForLoopBlock(page, /loop/i, '{{trigger.items}}');

    // 7. CONFIGURE LLM: Use the loop item variable
    await configureLLMBlock(
      page,
      /llm/i,
      'Process this item: {{loop.item}}. Respond with just "processed".'
    );

    // 8. CONNECT: Trigger -> For Loop
    await connectNodes(page, /trigger/i, /loop/i);

    // 9. CONNECT: For Loop (Loop handle) -> LLM
    await connectNodes(page, /loop/i, /llm/i);

    // 10. VERIFY EDGES CREATED
    const edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(2);

    // 11. RUN WORKFLOW (opens trigger event selection dialog)
    await runWorkflow(page);

    // 12. VERIFY TRIGGER EVENT DIALOG APPEARS
    const triggerDialog = page.getByRole('dialog', { name: /Select Trigger Event/i });
    await expect(triggerDialog).toBeVisible({ timeout: 10000 });

    // 13. VERIFY WEBHOOK OPTION IS SHOWN (confirms trigger and structure)
    const webhookOption = triggerDialog.getByText('Webhook');
    await expect(webhookOption).toBeVisible();

    // 14. CLOSE DIALOG (workflow structure validated)
    const closeButton = triggerDialog.getByRole('button', { name: /Close/i });
    await closeButton.click();

    console.log('For Loop workflow structure validated successfully');
  });

  test('for loop node shows correct handles', async ({ page }) => {
    // Add For Loop block
    await addBlock(page, 'for_loop');

    const forLoopNode = page.locator('.react-flow__node').filter({ hasText: /loop/i });
    await expect(forLoopNode).toBeVisible();

    // Verify Loop output handle label (exact match to avoid matching "loop-1" node label)
    const loopLabel = forLoopNode.getByText('Loop', { exact: true });
    await expect(loopLabel).toBeVisible();

    // Verify Exit output handle label
    const exitLabel = forLoopNode.getByText('Exit', { exact: true });
    await expect(exitLabel).toBeVisible();

    // Verify handles exist (input + 2 outputs)
    const handles = forLoopNode.locator('.react-flow__handle');
    const handleCount = await handles.count();
    expect(handleCount).toBeGreaterThanOrEqual(3); // 1 input + 2 outputs
  });
});

import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addTrigger,
  addBlock,
  connectNodes,
  runWorkflow,
  getNodeCount,
  getEdgeCount,
  organizeLayout,
} from '../../fixtures/utils';

test.describe('Conditional Workflow (If/Else)', () => {
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

  test('create workflow with if/else branching', async ({ page }) => {
    // 1. ADD TRIGGER: Webhook (no OAuth needed)
    await addTrigger(page, 'webhook.generic');
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();

    // 2. ADD IF/ELSE BLOCK
    await addBlock(page, 'if_else');
    const ifElseNode = page.locator('.react-flow__node').filter({ hasText: /if/i });
    await expect(ifElseNode).toBeVisible();

    // 3. VERIFY BRANCH LABELS EXIST
    await expect(ifElseNode.getByText('True')).toBeVisible();
    await expect(ifElseNode.getByText('False')).toBeVisible();

    // 4. ADD TWO LLM BLOCKS (one for each branch)
    await addBlock(page, 'llm'); // LLM for True branch
    await addBlock(page, 'llm'); // LLM for False branch

    // Verify we have 4 nodes: trigger + if/else + 2 LLMs
    const nodeCount = await getNodeCount(page);
    expect(nodeCount).toBe(4);

    // 5. ORGANIZE LAYOUT: Separate overlapping nodes before connecting
    await organizeLayout(page);

    // 6. CONNECT: Trigger -> If/Else
    await connectNodes(page, /trigger/i, /if/i);

    // 7. VERIFY EDGE CREATED
    const edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(1);

    // 8. RUN WORKFLOW (opens trigger event selection dialog)
    await runWorkflow(page);

    // 9. VERIFY TRIGGER EVENT DIALOG APPEARS
    // Webhook triggers require selecting a real event to test with
    const triggerDialog = page.getByRole('dialog', { name: /Select Trigger Event/i });
    await expect(triggerDialog).toBeVisible({ timeout: 10000 });

    // 10. VERIFY WEBHOOK OPTION IS SHOWN (confirms trigger was added correctly)
    const webhookOption = triggerDialog.getByText('Webhook');
    await expect(webhookOption).toBeVisible();

    // 11. CLOSE DIALOG (no events to select for testing)
    const closeButton = triggerDialog.getByRole('button', { name: /Close/i });
    await closeButton.click();

    // Workflow build is complete - execution requires real events
  });

  test('if/else node shows branch handles correctly', async ({ page }) => {
    // Add If/Else block
    await addBlock(page, 'if_else');

    const ifElseNode = page.locator('.react-flow__node').filter({ hasText: /if/i });
    await expect(ifElseNode).toBeVisible();

    // Verify True branch (orange handle)
    const trueLabel = ifElseNode.getByText('True');
    await expect(trueLabel).toBeVisible();

    // Verify False branch (muted handle)
    const falseLabel = ifElseNode.getByText('False');
    await expect(falseLabel).toBeVisible();

    // Verify multiple output handles exist
    const handles = ifElseNode.locator('.react-flow__handle');
    const handleCount = await handles.count();
    expect(handleCount).toBeGreaterThanOrEqual(3); // 1 input + 2 outputs (true/false)
  });
});

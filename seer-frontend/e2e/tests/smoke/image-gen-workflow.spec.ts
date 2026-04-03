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
  configureImageGenBlock,
  runWorkflow,
  selectNode,
  switchBuildPanelTab,
} from '../../fixtures/utils';

test.describe('Image Gen Workflow', () => {
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

  test('create workflow with form trigger and image gen', async ({ page }) => {
    // 1. ADD TRIGGER: Hosted Form
    await addTrigger(page, 'form.hosted');
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();

    // 2. ADD IMAGE GEN BLOCK
    await addBlock(page, 'image_gen');
    const imageGenNode = page.locator('.react-flow__node').filter({ hasText: /img|image/i });
    await expect(imageGenNode).toBeVisible();

    // Verify we have 2 nodes: trigger + image_gen
    const nodeCount = await getNodeCount(page);
    expect(nodeCount).toBe(2);

    // 3. ORGANIZE LAYOUT before configuring
    await organizeLayout(page);

    // 4. CONFIGURE IMAGE GEN: Set prompt using form field reference
    await configureImageGenBlock(
      page,
      /img|image/i,
      '{{trigger.form_data.prompt}}'
    );

    // 5. CONNECT: Form Trigger -> Image Gen
    await connectNodes(page, /trigger/i, /img|image/i);

    // 6. VERIFY EDGE CREATED
    const edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(1);

    // 7. RUN WORKFLOW (opens trigger event selection dialog)
    await runWorkflow(page);

    // 8. VERIFY TRIGGER EVENT DIALOG APPEARS
    const triggerDialog = page.getByRole('dialog', { name: /Select Trigger Event/i });
    await expect(triggerDialog).toBeVisible({ timeout: 10000 });

    // 9. VERIFY FORM OPTION IS SHOWN (confirms trigger configured)
    const formOption = triggerDialog.getByText('Form');
    await expect(formOption).toBeVisible();

    // 10. CLOSE DIALOG
    const closeButton = triggerDialog.getByRole('button', { name: /Close/i });
    await closeButton.click();

    console.log('Image Gen workflow with form trigger validated successfully');
  });

  test('image gen node shows correct configuration fields', async ({ page }) => {
    // Add Image Gen block
    await addBlock(page, 'image_gen');

    const imageGenNode = page.locator('.react-flow__node').filter({ hasText: /img|image/i });
    await expect(imageGenNode).toBeVisible();

    // Select the node and check config panel
    await selectNode(page, /img|image/i);
    await switchBuildPanelTab(page, 'Config');

    // Verify config fields are visible (label is "Prompt*" for required fields)
    const promptField = page.getByRole('textbox', { name: /^Prompt\*?$/i });
    await expect(promptField).toBeVisible({ timeout: 5000 });

    // Verify Model dropdown exists (combobox role)
    const modelField = page.getByRole('combobox', { name: /Model/i });
    await expect(modelField).toBeVisible();

    // Verify Size dropdown exists
    const sizeField = page.getByRole('combobox', { name: /Size/i });
    await expect(sizeField).toBeVisible();
  });

  test('workflow with webhook trigger and image gen', async ({ page }) => {
    // 1. ADD TRIGGER: Webhook
    await addTrigger(page, 'webhook.generic');
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();

    // 2. ADD IMAGE GEN BLOCK
    await addBlock(page, 'image_gen');
    await expect(page.locator('.react-flow__node').filter({ hasText: /img|image/i })).toBeVisible();

    // 3. ORGANIZE LAYOUT
    await organizeLayout(page);

    // 4. CONFIGURE IMAGE GEN with webhook payload reference
    await configureImageGenBlock(
      page,
      /img|image/i,
      '{{trigger.prompt}}'
    );

    // 5. CONNECT nodes
    await connectNodes(page, /trigger/i, /img|image/i);

    // 6. VERIFY EDGE CREATED
    const edgeCount = await getEdgeCount(page);
    expect(edgeCount).toBeGreaterThanOrEqual(1);

    // 7. RUN WORKFLOW
    await runWorkflow(page);

    // 8. VERIFY TRIGGER EVENT DIALOG
    const triggerDialog = page.getByRole('dialog', { name: /Select Trigger Event/i });
    await expect(triggerDialog).toBeVisible({ timeout: 10000 });

    // 9. VERIFY WEBHOOK OPTION
    const webhookOption = triggerDialog.getByText('Webhook');
    await expect(webhookOption).toBeVisible();

    // 10. CLOSE DIALOG
    const closeButton = triggerDialog.getByRole('button', { name: /Close/i });
    await closeButton.click();

    console.log('Image Gen workflow with webhook trigger validated successfully');
  });
});

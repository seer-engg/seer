/**
 * Smoke Test: Agent Workflow Creation + Execution
 *
 * Validates the core product loop:
 * 1. Create workflow with webhook trigger + agent block (web_search)
 * 2. Configure agent with instructions
 * 3. Publish workflow
 * 4. Trigger via webhook
 * 5. Assert execution succeeds
 */
import { test, expect, type Page } from '../../fixtures/auth';
import {
  deleteTestWorkflow,
  createTestWorkflow,
  getClerkToken,
  getWebhookSubscription,
  postToWebhook,
} from '../../fixtures/utils';

/** Add webhook trigger + agent block to an empty canvas. */
async function addTriggerAndAgent(page: Page) {
  const addFirstBlock = page.getByRole('button', { name: 'Add first block' });
  await expect(addFirstBlock).toBeVisible({ timeout: 30_000 });
  await addFirstBlock.click();

  const blockPickerDialog = page.getByRole('dialog', { name: /Select a block to add/i });
  await expect(blockPickerDialog).toBeVisible({ timeout: 5_000 });
  // Switch to Triggers tab (Webhook is a trigger, not on default Blocks & Tools tab)
  await blockPickerDialog.getByRole('tab', { name: /Triggers/i }).click();
  await blockPickerDialog.getByRole('button', { name: 'Webhook' }).click();
  await expect(page.getByText('trigger-1')).toBeVisible({ timeout: 10_000 });

  const addAfterTrigger = page.getByRole('button', { name: /Add node after trigger/i });
  await expect(addAfterTrigger).toBeVisible({ timeout: 5_000 });
  await addAfterTrigger.click();

  const addBlockDialog = page.getByRole('dialog', { name: /Select a block to add/i });
  await expect(addBlockDialog).toBeVisible({ timeout: 5_000 });
  await addBlockDialog.getByText('Agent', { exact: true }).click();
  await expect(page.getByText('agent-1')).toBeVisible({ timeout: 10_000 });
}

/** Open agent config dialog, fill instructions, verify websearch, close. */
async function configureAgent(page: Page) {
  const agentNodeEl = page.locator('.react-flow__node').filter({ hasText: 'agent-1' });
  await agentNodeEl.click();
  await page.waitForTimeout(500);

  const configDialog = page.getByRole('dialog', { name: /Node Config/i });
  await expect(configDialog).toBeVisible({ timeout: 5_000 });

  const instructionsField = configDialog.getByRole('textbox', { name: /Agent Instructions/i });
  await expect(instructionsField).toBeVisible();

  // Set up PATCH listener BEFORE typing so we catch the liveUpdate debounce save
  const draftSaved = page.waitForResponse(
    (resp) => resp.url().includes('/draft') && resp.request().method() === 'PATCH' && resp.status() === 200,
    { timeout: 15_000 }
  );

  await instructionsField.click();
  await instructionsField.pressSequentially(
    'Use web_search to find the latest news about artificial intelligence. Return a brief 2-sentence summary.',
    { delay: 5 }
  );

  // Wait for liveUpdate debounce (350ms) + PATCH to complete before closing
  await draftSaved;

  const websearchTool = configDialog.getByRole('button', { name: /Websearch/i });
  await expect(websearchTool).toBeVisible();
  await expect(websearchTool.getByText('Connected')).toBeVisible();

  await configDialog.getByRole('button', { name: /Close/i }).click();
  await expect(configDialog).not.toBeVisible({ timeout: 3_000 });
}

/** Fetch workflow JSON from the API. */
async function fetchWorkflow(page: Page, workflowId: string, token: string) {
  const baseUrl = process.env.VITE_BACKEND_API_URL || 'http://localhost:8000';
  const resp = await page.request.get(`${baseUrl}/api/v1/workflows/${workflowId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  expect(resp.ok()).toBeTruthy();
  return resp.json();
}

/** Get agent prompt from workflow data. */
function getAgentPrompt(wfData: Record<string, unknown>) {
  const nodes = ((wfData.spec as Record<string, unknown>)?.nodes || []) as Record<string, unknown>[];
  const agentNode = nodes.find((n) => n.id === 'agent-1');
  const inputs = (agentNode?.inputs ?? agentNode?.config) as Record<string, unknown> | undefined;
  return inputs?.prompt;
}

/** Fetch workflow, verify agent prompt is saved, return triggerId. */
async function verifyAgentConfig(page: Page, workflowId: string) {
  const token = await getClerkToken(page);
  const wfData = await fetchWorkflow(page, workflowId, token);
  const triggerId = (wfData.spec as Record<string, unknown>)?.triggers as Record<string, unknown>[] | undefined;
  const tid = (triggerId?.[0] as Record<string, unknown>)?.id;
  expect(tid).toBeTruthy();

  let prompt = getAgentPrompt(wfData);
  if (!prompt) {
    await page.waitForTimeout(3000);
    const retryData = await fetchWorkflow(page, workflowId, token);
    prompt = getAgentPrompt(retryData);
  }
  expect(prompt).toBeTruthy();

  return { token, triggerId: tid };
}

/** Poll page for execution terminal status by reloading periodically. */
async function pollForExecution(page: Page, timeoutMs = 120_000): Promise<string> {
  await page.waitForTimeout(2000);
  await page.reload();
  await page.waitForTimeout(3000);

  const runsButton = page.getByRole('button', { name: 'Runs' });
  if (await runsButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    await runsButton.click();
    await page.waitForTimeout(1000);
  }

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await page.getByText(/Succeeded/i).isVisible({ timeout: 1000 }).catch(() => false)) return 'succeeded';
    if (await page.getByText(/Failed/i).isVisible({ timeout: 1000 }).catch(() => false)) return 'failed';
    await page.waitForTimeout(5000);
    await page.reload();
    await page.waitForTimeout(2000);
  }
  return 'timeout';
}

test.describe('Agent Workflow: Create + Execute', () => {
  let workflowId: string | undefined;

  test.afterEach(async ({ page }) => {
    if (workflowId) {
      await deleteTestWorkflow(page, workflowId);
    }
  });

  test('create workflow with agent block, publish, trigger webhook, verify success', async ({ page }) => {
    test.setTimeout(180_000);

    // Step 1: Create workflow and navigate
    workflowId = await createTestWorkflow(page, `e2e-agent-smoke-${Date.now()}`);
    await page.goto(`/workflows/${workflowId}`);

    // Step 2-3: Add blocks
    await addTriggerAndAgent(page);

    // Step 4: Configure agent (PATCH save happens inside configureAgent before dialog closes)
    await configureAgent(page);

    // Step 5: Verify config persisted, provision webhook
    const { triggerId } = await verifyAgentConfig(page, workflowId!);
    const subscription = await getWebhookSubscription(page, workflowId!, triggerId);
    expect(subscription.webhookUrl).toBeTruthy();

    // Step 6: Publish
    const publishResp = page.waitForResponse(
      (resp) => resp.url().includes('/publish') && resp.status() === 200
    );
    await page.getByTestId('publish-button').click();
    await publishResp;
    await expect(page.getByText(/Workflow published/i)).toBeVisible({ timeout: 10_000 });

    // Step 7: Fire webhook and verify execution succeeds
    await postToWebhook(page, subscription.webhookUrl, { query: 'latest AI news' });
    const status = await pollForExecution(page);
    expect(status).toBe('succeeded');
  });
});

/**
 * E2E Test Utilities
 *
 * Shared helpers for Playwright tests to handle common operations
 * like navigation, onboarding dismissal, and UI interactions.
 */
import { expect } from '@playwright/test';
import type { Page, Locator } from '@playwright/test';

/**
 * Get Clerk auth token from the browser context.
 * Must be called after navigating to the app.
 */
export async function getClerkToken(page: Page): Promise<string> {
  // Wait for Clerk to be available
  await page.waitForFunction(
    () => {
      const w = window as unknown as { Clerk?: { session?: unknown } };
      return w.Clerk && w.Clerk.session;
    },
    { timeout: 10000 }
  );

  // Get the token - using a simple function that Playwright can serialize
  const token = await page.evaluate(() => {
    const w = window as unknown as { Clerk: { session: { getToken: () => Promise<string> } } };
    return w.Clerk.session.getToken();
  });

  return token;
}

/**
 * Create a fresh workflow for test isolation.
 * Uses the backend API to create an empty workflow with a unique name.
 * This ensures each test has its own workflow and tests can run in parallel.
 */
export async function createTestWorkflow(page: Page, name?: string): Promise<string> {
  const workflowName = name || `e2e-test-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

  // First navigate to the app so Clerk is initialized
  await page.goto('/');

  // Wait for the app to load
  await page.waitForSelector('main', { timeout: 15000 }).catch(() => {});
  await page.waitForTimeout(500);

  // Get Clerk auth token
  const token = await getClerkToken(page);

  // Get backend URL from environment or default
  const baseUrl = process.env.VITE_BACKEND_API_URL || 'http://localhost:8000';

  // Create workflow via API with explicit auth header
  const response = await page.request.post(`${baseUrl}/api/v1/workflows`, {
    headers: {
      'Authorization': `Bearer ${token}`
    },
    data: {
      name: workflowName,
      spec: {
        version: '2',
        nodes: [],
        edges: [],
        triggers: []
      }
    }
  });

  if (!response.ok()) {
    const text = await response.text();
    throw new Error(`Failed to create workflow: ${response.status()} - ${text}`);
  }

  const data = await response.json();
  // API returns workflow_id, not id
  return data.workflow_id;
}

/**
 * Delete a test workflow for cleanup.
 * Silently ignores errors (workflow may already be deleted).
 */
export async function deleteTestWorkflow(page: Page, workflowId: string): Promise<void> {
  try {
    // Get Clerk auth token (may fail if page has navigated away)
    const token = await getClerkToken(page).catch(() => '');

    if (!token) {
      return; // Can't delete without auth
    }

    const baseUrl = process.env.VITE_BACKEND_API_URL || 'http://localhost:8000';

    await page.request.delete(`${baseUrl}/api/v1/workflows/${workflowId}`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }).catch(() => {});
  } catch {
    // Ignore deletion errors - workflow may already be deleted or test may have failed
  }
}

/**
 * Navigate to a workflow and wait for the canvas to load.
 * If workflowId is provided, navigates directly to that workflow.
 * Otherwise, creates a new workflow via API for test isolation.
 *
 * @returns The workflow ID (useful for cleanup in afterEach)
 */
export async function openWorkflow(page: Page, workflowId?: string): Promise<string> {
  let id = workflowId;

  // Create a new workflow if no ID provided (test isolation)
  if (!id) {
    id = await createTestWorkflow(page);
  }

  // Navigate directly to the workflow
  await page.goto(`/workflows/${id}`);

  // Wait for canvas to load
  await page.waitForSelector('[data-testid="workflow-canvas"]', { timeout: 30000 });

  return id;
}

/**
 * Add a block to the canvas by clicking it in the Build panel.
 * Waits for the debounce cycle to ensure state is properly saved.
 * @param blockType - One of: llm, if_else, for_loop, mcp, hitl, browser, image_gen
 */
export async function addBlock(page: Page, blockType: string): Promise<void> {
  // Ensure Build tab is active (might be on Config tab after configuring)
  await switchBuildPanelTab(page, 'Build');

  const blockSelector = page.locator(`[data-testid="block-${blockType}"]`);
  // Wait for block to be visible before clicking
  await blockSelector.waitFor({ state: 'visible', timeout: 5000 });
  await blockSelector.click();

  // Wait for debounce cycle to complete (1000ms debounce + buffer)
  // This ensures the save is triggered before adding the next block
  await page.waitForTimeout(2000);
}

/**
 * Add a trigger to the canvas.
 * Waits for the debounce cycle to ensure state is properly saved.
 * @param triggerKey - One of: webhook.generic, poll.gmail.email_received, schedule.cron, etc.
 */
export async function addTrigger(page: Page, triggerKey: string): Promise<void> {
  // Ensure Build tab is active
  await switchBuildPanelTab(page, 'Build');

  const triggerSelector = page.locator(`[data-testid="trigger-${triggerKey}"]`);
  await triggerSelector.waitFor({ state: 'visible', timeout: 5000 });
  await triggerSelector.click();

  // Wait for debounce cycle to complete
  await page.waitForTimeout(2000);
}

/**
 * Search for items in the Build panel.
 */
export async function searchBuildPanel(page: Page, query: string): Promise<void> {
  const searchInput = page.locator('[data-testid="build-search"]');
  await searchInput.waitFor({ state: 'visible', timeout: 5000 });
  await searchInput.fill(query);
}

/**
 * Click the Run button to execute the workflow.
 */
export async function runWorkflow(page: Page): Promise<void> {
  const runButton = page.locator('[data-testid="run-button"]');
  await runButton.waitFor({ state: 'visible', timeout: 5000 });
  await runButton.click();
}

/**
 * Click the Publish button and wait for publish to complete.
 * Waits for either a success toast or a validation error toast to appear,
 * so callers don't need manual delays to know publish has settled.
 */
export async function publishWorkflow(page: Page): Promise<void> {
  const publishButton = page.locator('[data-testid="publish-button"]');
  await publishButton.waitFor({ state: 'visible', timeout: 5000 });
  await publishButton.click();
  // Wait for publish to complete (success or error toast appears within ~8 s)
  await expect(
    page.getByText(/Workflow published|validation failed/i)
  ).toBeVisible({ timeout: 8000 }).catch(() => {
    // Toast may not appear for all publish states — continue regardless
  });
}

/**
 * Wait for the canvas to be fully loaded and interactive.
 */
export async function waitForCanvasReady(page: Page): Promise<void> {
  await page.waitForSelector('[data-testid="workflow-canvas"]', { timeout: 10000 });
  // Wait for React Flow to initialize (control panel visible)
  await page.waitForSelector('.react-flow__controls', { timeout: 10000 });
}

/**
 * Wait for workflow data to load after a page refresh.
 * This waits for the workflow API response to complete and nodes to render.
 * Use this after page.reload() when you expect nodes to be present.
 * @param page - Playwright page
 * @param expectedMinNodes - Minimum number of nodes expected (default: 1)
 */
export async function waitForWorkflowLoaded(page: Page, expectedMinNodes = 1): Promise<void> {
  // First wait for canvas
  await waitForCanvasReady(page);

  // Wait for the workflow API call to complete by waiting for nodes to appear
  if (expectedMinNodes > 0) {
    await page.waitForFunction(
      (minNodes) => {
        const nodes = document.querySelectorAll('.react-flow__node');
        return nodes.length >= minNodes;
      },
      expectedMinNodes,
      { timeout: 15000 }
    );
  }
}

/**
 * Wait for autosave to complete.
 * The workflow uses a 1000ms debounce, so we wait for debounce + save time.
 * This is more reliable than looking for UI indicators since autosave is silent.
 */
export async function waitForAutosave(page: Page): Promise<void> {
  // Wait for debounce (1000ms) + network save time + generous buffer
  // Complex workflows with many nodes may take longer to serialize and save
  await page.waitForTimeout(3500);
}

/**
 * Select a node on the canvas by its label pattern.
 * Uses React Flow's fitView to ensure node is visible before clicking.
 * If the click doesn't select the node (due to overlapping nodes), automatically
 * organizes the layout and retries.
 *
 * Note: scrollIntoViewIfNeeded() doesn't work with React Flow because it uses
 * CSS transforms for panning/zooming, not native scroll. The fitView control
 * adjusts the viewport transform to show all nodes.
 *
 * @param labelPattern - A regex pattern or string to match the node label
 */
export async function selectNode(page: Page, labelPattern: string | RegExp): Promise<void> {
  const node = page.locator('.react-flow__node').filter({ hasText: labelPattern }).first();

  // Wait for node to exist in DOM
  await node.waitFor({ state: 'attached', timeout: 5000 });

  // Use the "Fit view" control button to ensure all nodes are visible
  // This is critical for off-canvas nodes that React Flow may position outside the viewport
  const fitViewButton = page.locator('.react-flow__controls-fitview');
  if (await fitViewButton.isVisible({ timeout: 1000 }).catch(() => false)) {
    await fitViewButton.click();
    await page.waitForTimeout(300); // Wait for animation
  }

  // Click the node - use force:true to bypass any overlay issues
  await node.click({ force: true });
  await page.waitForTimeout(100);

  // Check if node was actually selected (React Flow adds 'selected' class)
  const isSelected = await node.evaluate(el => el.classList.contains('selected'));

  if (!isSelected) {
    // Node wasn't selected - likely due to overlapping nodes
    // Organize layout to separate nodes and retry
    const organizeButton = page.getByRole('button', { name: /Organize layout/i });
    if (await organizeButton.isVisible({ timeout: 1000 }).catch(() => false)) {
      await organizeButton.click();
      await page.waitForTimeout(500); // Wait for layout animation
    }

    // Retry clicking the node
    await node.click({ force: true });
    await page.waitForTimeout(100);
  }
}

/**
 * Get the count of nodes on the canvas.
 */
export async function getNodeCount(page: Page): Promise<number> {
  return page.locator('.react-flow__node').count();
}

/**
 * Get the count of edges on the canvas.
 */
export async function getEdgeCount(page: Page): Promise<number> {
  return page.locator('.react-flow__edge').count();
}

/**
 * Switch to a specific tab in the Build panel.
 * @param tabName - One of: 'Build', 'Executions', 'Config'
 */
export async function switchBuildPanelTab(page: Page, tabName: 'Build' | 'Executions' | 'Config'): Promise<void> {
  const tab = page.getByRole('tab', { name: tabName });
  // Only click if not already selected to avoid unnecessary state changes
  const isSelected = await tab.getAttribute('aria-selected');
  if (isSelected !== 'true') {
    await tab.click();
    // Wait for tab content to render
    await page.waitForTimeout(100);
  }
}

/**
 * Organize the layout of nodes on the canvas using the auto-layout button.
 * This separates overlapping nodes so they can be connected properly.
 */
export async function organizeLayout(page: Page): Promise<void> {
  const organizeButton = page.getByRole('button', { name: /Organize layout/i });
  await organizeButton.waitFor({ state: 'visible', timeout: 5000 });
  await organizeButton.click();
  // Wait for the layout animation to complete
  await page.waitForTimeout(1000);
}

/**
 * Check if Gmail OAuth is connected by looking for the "Connected" badge
 * on the Gmail trigger card in the Build panel.
 */
export async function isGmailConnected(page: Page): Promise<boolean> {
  // Find the Gmail trigger card by its exact data-testid
  const gmailTrigger = page.locator('[data-testid="trigger-poll.gmail.email_received"]');

  // Check if the trigger card is visible
  if (!await gmailTrigger.isVisible({ timeout: 3000 }).catch(() => false)) {
    return false;
  }

  // Check if "Connected" badge exists WITHIN this specific trigger card
  const connectedBadge = gmailTrigger.getByText('Connected');
  return await connectedBadge.isVisible({ timeout: 1000 }).catch(() => false);
}

/**
 * Connect two nodes by dragging from source output handle to target input handle.
 * Uses force: true to bypass any overlapping element issues.
 */
export async function connectNodes(
  page: Page,
  sourceNodeLabel: string | RegExp,
  targetNodeLabel: string | RegExp
): Promise<void> {
  // Find source node's output handle
  const sourceNode = page.locator('.react-flow__node').filter({ hasText: sourceNodeLabel });
  const sourceHandle = sourceNode.locator('.react-flow__handle[data-handlepos="bottom"], .react-flow__handle[data-handlepos="right"]').first();

  // Find target node's input handle
  const targetNode = page.locator('.react-flow__node').filter({ hasText: targetNodeLabel });
  const targetHandle = targetNode.locator('.react-flow__handle[data-handlepos="top"], .react-flow__handle[data-handlepos="left"]').first();

  // Drag to connect using force to bypass any overlay issues
  await sourceHandle.hover({ force: true });
  await page.mouse.down();
  await targetHandle.hover({ force: true });
  await page.mouse.up();

  // Wait for edge to be created
  await page.waitForTimeout(500);
}

/**
 * Wait for workflow execution to complete by polling the Executions tab.
 */
export async function waitForExecutionComplete(
  page: Page,
  options: { timeout?: number } = {}
): Promise<void> {
  const timeout = options.timeout || 30000;
  const startTime = Date.now();

  // Switch to Executions tab
  await switchBuildPanelTab(page, 'Executions');

  // Poll for completion status
  while (Date.now() - startTime < timeout) {
    // Look for completion indicators
    const completed = page.getByText(/Completed|Success|finished/i);
    const failed = page.getByText(/Failed|Error/i);

    if (await completed.isVisible({ timeout: 1000 }).catch(() => false)) {
      return; // Success
    }
    if (await failed.isVisible({ timeout: 500 }).catch(() => false)) {
      return; // Failed but execution completed
    }

    await page.waitForTimeout(1000); // Poll every second
  }

  throw new Error(`Execution did not complete within ${timeout}ms`);
}

/**
 * Configure an LLM block with a prompt.
 */
export async function configureLLMBlock(
  page: Page,
  nodeLabel: string | RegExp,
  prompt: string
): Promise<void> {
  // Select the node
  await selectNode(page, nodeLabel);

  // Switch to Config tab
  await switchBuildPanelTab(page, 'Config');

  // Find and fill prompt field (use textbox role to avoid matching tooltip buttons)
  const promptField = page.getByRole('textbox', { name: /Prompt/i });
  if (await promptField.isVisible({ timeout: 3000 }).catch(() => false)) {
    await promptField.fill(prompt);
  }

  // Wait for autosave
  await page.waitForTimeout(1000);
}

/**
 * Configure an If/Else block with a condition.
 */
export async function configureIfElseBlock(
  page: Page,
  nodeLabel: string | RegExp,
  condition: string
): Promise<void> {
  // Select the node
  await selectNode(page, nodeLabel);

  // Switch to Config tab
  await switchBuildPanelTab(page, 'Config');

  // Find and fill condition field
  const conditionField = page.getByRole('textbox', { name: /condition/i });
  if (await conditionField.isVisible({ timeout: 3000 }).catch(() => false)) {
    await conditionField.fill(condition);
  }

  // Wait for autosave
  await page.waitForTimeout(1000);
}

/**
 * Select the first (latest) event from the TriggerEventPicker dialog.
 * Returns false if no events are available.
 */
export async function selectTriggerEvent(
  page: Page,
  options: { timeout?: number } = {}
): Promise<boolean> {
  const { timeout = 15000 } = options;
  const dialog = page.getByRole('dialog');

  // Wait for dialog to be visible
  await expect(dialog).toBeVisible({ timeout });

  // Check all "no events" states:
  // - TriggerEventList empty state: "No events found"
  // - Gmail-specific: "No emails"
  // - TriggerEventPicker shouldDisableFetcher: "No events received yet" / "Send a test event"
  // - RunOptionsDialog DisconnectedTriggerItem: "Send a test event to browse real events"
  const noEventPatterns = [
    /No events found/i,
    /No emails/i,
    /No events received yet/i,
    /Send a test event/i,
  ];
  for (const pattern of noEventPatterns) {
    if (await dialog.getByText(pattern).isVisible({ timeout: 1500 }).catch(() => false)) {
      return false;
    }
  }

  // Check for API error state (TriggerEventList shows a Retry button)
  if (await dialog.getByRole('button', { name: /Retry/i }).isVisible({ timeout: 1000 }).catch(() => false)) {
    return false;
  }

  // Find event items using semantic test id (set in TriggerEventListItem)
  const eventItem = dialog.getByTestId('trigger-event-item').first();
  await expect(eventItem).toBeVisible({ timeout: 10000 });

  await eventItem.click();

  // Wait for dialog to close (execution starts)
  try { await expect(dialog).not.toBeVisible({ timeout: 5000 }); } catch { /* dialog may stay open */ }

  return true;
}

/**
 * Wait for execution to reach terminal status and return it.
 * Polls the Executions tab for Succeeded/Failed badges.
 */
export async function waitForExecutionStatus(
  page: Page,
  options: { timeout?: number } = {}
): Promise<'succeeded' | 'failed' | 'interrupted' | 'timeout'> {
  const { timeout = 120000 } = options; // 2 minutes default for LLM workflows
  const startTime = Date.now();

  await switchBuildPanelTab(page, 'Executions');

  // Wait for execution list to appear (at least one run item)
  // Execution items are <button> elements — they do not have a literal `cursor-pointer` class
  const executionItem = page.locator('button').filter({ hasText: /Running|Succeeded|Failed|Queued|Awaiting/i });
  await executionItem.first().waitFor({ state: 'visible', timeout: 30000 }).catch(() => {});

  // Scope all status checks to the active tab panel to avoid matching hidden elements
  // (e.g. canvas node badges or other tab panels that also contain status text)
  const panel = page.locator('[role="tabpanel"][data-state="active"]');

  while (Date.now() - startTime < timeout) {
    // Check for Succeeded badge (case-insensitive)
    const succeeded = panel.getByText(/Succeeded/i);
    if (await succeeded.isVisible({ timeout: 2000 }).catch(() => false)) {
      return 'succeeded';
    }

    // Check for Failed badge (case-insensitive)
    const failed = panel.getByText(/Failed/i);
    if (await failed.isVisible({ timeout: 1000 }).catch(() => false)) {
      return 'failed';
    }

    // Check for Awaiting Input badge (HITL interrupted status)
    const interrupted = panel.getByText(/Awaiting Input/i);
    if (await interrupted.isVisible({ timeout: 1000 }).catch(() => false)) {
      return 'interrupted';
    }

    // Check if still running - continue polling
    const running = panel.getByText(/Running/i);
    const isRunning = await running.isVisible({ timeout: 500 }).catch(() => false);

    if (isRunning) {
      // Still running, wait and continue
      await page.waitForTimeout(3000);
      continue;
    }

    await page.waitForTimeout(2000);
  }

  return 'timeout';
}

// ============================================================================
// FOR LOOP CONFIGURATION
// ============================================================================

/**
 * Configure a For Loop block with an array source variable.
 */
export async function configureForLoopBlock(
  page: Page,
  nodeLabel: string | RegExp,
  arrayVariable: string
): Promise<void> {
  // Select the node
  await selectNode(page, nodeLabel);

  // Switch to Config tab
  await switchBuildPanelTab(page, 'Config');

  // Find and fill the array_variable field (use textbox role to avoid matching tooltip button)
  const arrayInput = page.getByRole('textbox', { name: /Array source variable/i });
  await arrayInput.waitFor({ state: 'visible', timeout: 5000 });
  await arrayInput.fill(arrayVariable);

  // Wait for autosave
  await page.waitForTimeout(1000);
}

// ============================================================================
// TOOL BLOCKS
// ============================================================================

/**
 * Add a tool from the Build panel by searching for it.
 * @param toolName - The name of the tool to search for (e.g., "discord_send_message")
 */
export async function addTool(page: Page, toolName: string): Promise<void> {
  // Search for the tool in the Build panel
  await searchBuildPanel(page, toolName);

  // Wait for search results
  await page.waitForTimeout(500);

  // Click the matching tool item
  const toolItem = page.locator('[data-testid="build-panel"]')
    .locator('[class*="cursor-pointer"]')
    .filter({ hasText: new RegExp(toolName, 'i') })
    .first();

  await toolItem.waitFor({ state: 'visible', timeout: 5000 });
  await toolItem.click();

  // Clear search
  const searchInput = page.locator('[data-testid="build-search"]');
  await searchInput.fill('');
}

/**
 * Get the webhook URL from a published workflow's trigger ui_meta.
 * After publish, the backend enriches triggers with webhook_url in ui_meta.
 * No need for start-listening — just read the published workflow.
 */
export async function getPublishedWebhookUrl(
  page: Page,
  workflowId: string,
  triggerId: string
): Promise<string> {
  const token = await getClerkToken(page);
  const baseUrl = process.env.VITE_BACKEND_API_URL || 'http://localhost:8000';
  const resp = await page.request.get(`${baseUrl}/api/v1/workflows/${workflowId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!resp.ok()) throw new Error(`Failed to get workflow: ${resp.status()}`);
  const data = await resp.json();
  const trigger = (data.spec?.triggers || []).find((t: Record<string, unknown>) => t.id === triggerId);
  const webhookPath = trigger?.ui_meta?.webhook_url;
  if (!webhookPath) throw new Error('No webhook_url in trigger ui_meta — is workflow published?');
  return `${baseUrl}${webhookPath}`;
}

// ============================================================================
// WEBHOOK TRIGGERS
// ============================================================================

/**
 * Webhook subscription data returned from startListening
 */
interface WebhookSubscription {
  webhookUrl: string;
  postUrl: string;  // URL to POST events to (may differ from display URL)
  secretToken: string;
  subscriptionId: number;
}

/**
 * Start listening on a webhook trigger and get the webhook URL and secret token.
 * Uses the startListening API which provisions the webhook without full publish.
 */
export async function getWebhookSubscription(
  page: Page,
  workflowId: string,
  triggerId: string
): Promise<WebhookSubscription> {
  const token = await getClerkToken(page);
  const baseUrl = process.env.VITE_BACKEND_API_URL || 'http://localhost:8000';

  const response = await page.request.post(
    `${baseUrl}/api/v1/workflows/${workflowId}/triggers/${triggerId}/start-listening`,
    {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );

  if (!response.ok()) {
    const text = await response.text();
    throw new Error(`Failed to start listening: ${response.status()} - ${text}`);
  }

  const data = await response.json();
  const webhookUrl = data.webhook_url;
  const secretToken = data.secret_token;
  const subscriptionId = data.subscription_id;

  if (!webhookUrl && !subscriptionId) {
    throw new Error('No webhook URL or subscription ID returned from startListening');
  }

  // Extract the path from the webhook URL and rebuild with correct baseUrl.
  // The backend may return a full URL with a different host (e.g., prod webhook_base_url),
  // so always rewrite to use our baseUrl for local/CI environments.
  let postUrl: string;
  if (webhookUrl) {
    const match = webhookUrl.match(/(\/api\/v1\/webhooks\/.*)/);
    const path = match ? match[1] : webhookUrl;
    postUrl = path.startsWith('/') ? `${baseUrl}${path}` : `${baseUrl}/${path}`;
  } else {
    postUrl = `${baseUrl}/api/v1/webhooks/generic/${subscriptionId}`;
  }

  return {
    webhookUrl: postUrl,
    postUrl,
    secretToken: secretToken || '',
    subscriptionId
  };
}

/**
 * Legacy function for backward compatibility - returns just the display URL
 */
export async function getWebhookUrl(page: Page, workflowId: string, triggerId: string): Promise<string> {
  const subscription = await getWebhookSubscription(page, workflowId, triggerId);
  return subscription.webhookUrl;
}

/**
 * Get the URL to POST webhook events to.
 * This may differ from the display URL (slug-based vs integer-based).
 */
export async function getWebhookPostUrl(page: Page, workflowId: string, triggerId: string): Promise<string> {
  const subscription = await getWebhookSubscription(page, workflowId, triggerId);
  return subscription.postUrl;
}

/**
 * Post a payload to a webhook URL to trigger workflow execution.
 * Simplified webhook system uses URL-based authentication (slug in URL).
 * No signing secret or Authorization header required.
 */
export async function postToWebhook(
  page: Page,
  webhookUrl: string,
  payload: Record<string, unknown>
): Promise<void> {
  // Simplified webhooks use URL-based security (slug acts as auth)
  // No Authorization header needed
  const response = await page.request.post(webhookUrl, {
    data: payload,
    headers: {
      'Content-Type': 'application/json'
    }
  });

  if (!response.ok()) {
    const text = await response.text();
    throw new Error(`Webhook POST failed: ${response.status()} - ${text}`);
  }
}

// ============================================================================
// IMAGE GEN CONFIGURATION
// ============================================================================

/**
 * Configure an Image Gen block with a prompt.
 */
export async function configureImageGenBlock(
  page: Page,
  nodeLabel: string | RegExp,
  prompt: string
): Promise<void> {
  // Select the node
  await selectNode(page, nodeLabel);

  // Switch to Config tab
  await switchBuildPanelTab(page, 'Config');

  // Find and fill the prompt field (label is "Prompt*" for required fields)
  const promptField = page.getByRole('textbox', { name: /^Prompt\*?$/i });
  await promptField.waitFor({ state: 'visible', timeout: 5000 });
  await promptField.fill(prompt);

  // Wait for autosave
  await page.waitForTimeout(1000);
}

// ============================================================================
// FORM TRIGGERS
// ============================================================================

/**
 * Get the form URL for a hosted form trigger.
 * Uses startListening API to provision the form.
 */
export async function getFormUrl(page: Page, workflowId: string, triggerId: string): Promise<string> {
  const token = await getClerkToken(page);
  const baseUrl = process.env.VITE_BACKEND_API_URL || 'http://localhost:8000';

  const response = await page.request.post(
    `${baseUrl}/api/v1/workflows/${workflowId}/triggers/${triggerId}/start-listening`,
    {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );

  if (!response.ok()) {
    const text = await response.text();
    throw new Error(`Failed to start listening: ${response.status()} - ${text}`);
  }

  const data = await response.json();
  const formUrl = data.form_url;

  if (!formUrl) {
    throw new Error('No form URL returned from startListening');
  }

  return formUrl;
}

/**
 * Submit form data to a hosted form trigger URL.
 */
export async function submitFormTrigger(
  page: Page,
  formUrl: string,
  formData: Record<string, unknown>
): Promise<void> {
  const response = await page.request.post(formUrl, {
    data: formData,
    headers: {
      'Content-Type': 'application/json'
    }
  });

  if (!response.ok()) {
    const text = await response.text();
    throw new Error(`Form submission failed: ${response.status()} - ${text}`);
  }
}

// ============================================================================
// HITL CONFIGURATION
// ============================================================================

/**
 * Configure a HITL (Human-in-the-Loop) block.
 */
export async function configureHITLBlock(
  page: Page,
  nodeLabel: string | RegExp,
  config: {
    title: string;
    description?: string;
  }
): Promise<void> {
  // Select the node
  await selectNode(page, nodeLabel);

  // Switch to Config tab
  await switchBuildPanelTab(page, 'Config');

  // Fill title (use textbox role to avoid matching tooltip buttons, label is "Title*" for required)
  const titleField = page.getByRole('textbox', { name: /^Title\*?$/i });
  await titleField.waitFor({ state: 'visible', timeout: 5000 });
  await titleField.fill(config.title);

  // Fill description if provided
  if (config.description) {
    const descField = page.getByRole('textbox', { name: /^Description$/i });
    await descField.fill(config.description);
  }

  // Wait for autosave
  await page.waitForTimeout(1000);
}

/**
 * Wait for a workflow execution to reach HITL interrupted state.
 * Returns the run ID of the interrupted execution.
 */
export async function waitForHITLPending(
  page: Page,
  options: { timeout?: number } = {}
): Promise<string | null> {
  const { timeout = 60000 } = options;

  const status = await waitForExecutionStatus(page, { timeout });

  if (status === 'interrupted') {
    // Get the run ID from the first execution item
    const runIdElement = page.locator('.font-mono').filter({ hasText: /^[a-f0-9-]+$/i }).first();
    const runId = await runIdElement.textContent();
    return runId?.trim() || null;
  }

  return null;
}

/**
 * Click on an execution to open its detail page, then approve the HITL prompt.
 */
export async function approveHITL(
  page: Page,
  runId: string
): Promise<void> {
  // Navigate to execution detail page
  await page.goto(`/executions/${runId}`);

  // Wait for the HITL response panel to appear
  const submitButton = page.getByRole('button', { name: /Submit Response/i });
  await submitButton.waitFor({ state: 'visible', timeout: 15000 });

  // Find and select "Approve" option if it's a single_choice field
  const approveOption = page.getByLabel(/Approve/i);
  if (await approveOption.isVisible({ timeout: 2000 }).catch(() => false)) {
    await approveOption.click();
  }

  // Click Submit Response
  await submitButton.click();

  // Wait for submission to complete (button should disappear or show success)
  await page.waitForTimeout(2000);
}

/**
 * Get the first trigger ID from a workflow's triggers.
 * Reads from the trigger node on the canvas.
 */
export async function getFirstTriggerId(page: Page, workflowId: string): Promise<string> {
  if (!workflowId || workflowId === 'undefined' || workflowId === 'null') {
    throw new Error(`Invalid workflowId passed to getFirstTriggerId: "${workflowId}"`);
  }

  const token = await getClerkToken(page);
  const baseUrl = process.env.VITE_BACKEND_API_URL || 'http://localhost:8000';

  const response = await page.request.get(
    `${baseUrl}/api/v1/workflows/${workflowId}`,
    {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );

  if (!response.ok()) {
    const text = await response.text();
    throw new Error(`Failed to get workflow: ${response.status()} - ${text}`);
  }

  const data = await response.json();
  const triggers = data.spec?.triggers || [];

  if (triggers.length === 0) {
    throw new Error(`No triggers found in workflow. Spec: ${JSON.stringify(data.spec)}`);
  }

  return triggers[0].id;
}

/**
 * Check if a tool is connected (OAuth configured) by looking for "Connected" badge.
 */
export async function isToolConnected(page: Page, toolName: string): Promise<boolean> {
  // Search for the tool
  await searchBuildPanel(page, toolName);
  await page.waitForTimeout(500);

  // Look for "Connected" badge on the tool item
  const toolItem = page.locator('[data-testid="build-panel"]')
    .locator('[class*="cursor-pointer"]')
    .filter({ hasText: new RegExp(toolName, 'i') })
    .first();

  if (!await toolItem.isVisible({ timeout: 3000 }).catch(() => false)) {
    return false;
  }

  const connectedBadge = toolItem.getByText('Connected');
  const isConnected = await connectedBadge.isVisible({ timeout: 1000 }).catch(() => false);

  // Clear search
  const searchInput = page.locator('[data-testid="build-search"]');
  await searchInput.fill('');

  return isConnected;
}

// ============================================================================
// VARIABLE AUTOCOMPLETE
// ============================================================================

/**
 * Inserts a variable using the autocomplete UI dropdown.
 * Types `{{` + filterText to trigger and filter suggestions, then clicks a match.
 *
 * This is more resilient than hardcoding variable paths because:
 * 1. Tests actual user behavior (using autocomplete)
 * 2. Adapts if variable names change in the backend
 * 3. Catches autocomplete UI bugs
 *
 * @param page - Playwright page
 * @param fieldLocator - The input/textarea locator to type into
 * @param filterText - Text to type after {{ to filter suggestions (e.g., "subj" for subject)
 * @param options.selectContaining - If provided, selects the suggestion whose text contains this substring
 */
export async function insertVariableViaAutocomplete(
  page: Page,
  fieldLocator: Locator,
  filterText: string,
  options?: { selectContaining?: string }
): Promise<void> {
  // Click field and type to trigger autocomplete
  await fieldLocator.click();
  await fieldLocator.pressSequentially(`{{${filterText}`, { delay: 50 });

  // Wait for autocomplete dropdown using semantic testid
  const dropdown = page.getByTestId('variable-autocomplete-dropdown');
  await expect(dropdown).toBeVisible({ timeout: 5000 });

  // Select matching suggestion by substring or fall back to first
  const items = dropdown.getByTestId('variable-suggestion-item');
  const suggestion = options?.selectContaining
    ? items.filter({ hasText: options.selectContaining }).first()
    : items.first();

  await expect(suggestion).toBeVisible({ timeout: 3000 });
  await suggestion.click();

  // Wait for autocomplete to insert and close
  await page.waitForTimeout(200);
}

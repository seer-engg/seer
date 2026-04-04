/**
 * Multi-Node Editing Isolation Tests
 *
 * Verifies that editing one node does not affect the configuration
 * of other nodes. This catches bugs in state management where node
 * configs might be accidentally overwritten or merged.
 */
import { test, expect } from '../../fixtures/auth';
import {
  openWorkflow,
  deleteTestWorkflow,
  waitForCanvasReady,
  addBlock,
  addTrigger,
  configureLLMBlock,
  configureIfElseBlock,
  selectNode,
  switchBuildPanelTab,
  organizeLayout,
} from '../../fixtures/utils';

test.describe('Multi-Node Editing Isolation', () => {
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

  test('editing Node B does not affect Node A configuration', async ({ page }) => {
    test.setTimeout(90000); // Multiple blocks with debounce waits + organize layout
    // Add and configure Node A (LLM-1)
    await addBlock(page, 'llm');
    await configureLLMBlock(page, /llm-1/i, 'First prompt: Alpha');

    // Add and configure Node B (LLM-2)
    await addBlock(page, 'llm');
    await configureLLMBlock(page, /llm-2/i, 'Second prompt: Beta');

    // Add and configure Node C (If/Else)
    await addBlock(page, 'if_else');
    await configureIfElseBlock(page, /if/i, '{{llm-1.output}} == "success"');

    // Wait for all autosaves to complete (debounce + buffer)
    await page.waitForTimeout(2500);

    // Go back to Node A and verify its config is intact
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    const prompt1 = page.getByRole('textbox', { name: /Prompt/i });
    await expect(prompt1).toHaveValue(/First prompt: Alpha/);

    // Verify Node B config is still intact
    await selectNode(page, /llm-2/i);
    await switchBuildPanelTab(page, 'Config');
    const prompt2 = page.getByRole('textbox', { name: /Prompt/i });
    await expect(prompt2).toHaveValue(/Second prompt: Beta/);
  });

  test('rapid switching between nodes preserves all configs', async ({ page }) => {
    test.setTimeout(120000); // Adding 4 blocks with debounce waits
    // Add 4 LLM blocks with distinct prompts
    await addBlock(page, 'llm'); // llm-1
    await addBlock(page, 'llm'); // llm-2
    await addBlock(page, 'llm'); // llm-3
    await addBlock(page, 'llm'); // llm-4

    // Configure each with unique prompts
    await configureLLMBlock(page, /llm-1/i, 'Prompt ONE');
    await configureLLMBlock(page, /llm-2/i, 'Prompt TWO');
    await configureLLMBlock(page, /llm-3/i, 'Prompt THREE');
    await configureLLMBlock(page, /llm-4/i, 'Prompt FOUR');

    // Wait for autosave
    await page.waitForTimeout(2000);

    // Rapidly switch between nodes multiple times
    for (let i = 0; i < 3; i++) {
      await selectNode(page, /llm-1/i);
      await page.waitForTimeout(100);
      await selectNode(page, /llm-2/i);
      await page.waitForTimeout(100);
      await selectNode(page, /llm-3/i);
      await page.waitForTimeout(100);
      await selectNode(page, /llm-4/i);
      await page.waitForTimeout(100);
    }

    // Verify all configs are still correct
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Prompt ONE/);

    await selectNode(page, /llm-2/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Prompt TWO/);

    await selectNode(page, /llm-3/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Prompt THREE/);

    await selectNode(page, /llm-4/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Prompt FOUR/);
  });

  test('editing config while another node is selected does not corrupt state', async ({ page }) => {
    test.setTimeout(60000); // Multiple selectNode calls may trigger organize layout
    // Add two LLM blocks
    await addBlock(page, 'llm'); // llm-1
    await addBlock(page, 'llm'); // llm-2

    // Configure llm-1
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    const promptInput = page.getByRole('textbox', { name: /Prompt/i });
    await promptInput.fill('LLM-1 Original');

    // Wait for autosave
    await page.waitForTimeout(1500);

    // Select llm-2 and configure it
    await selectNode(page, /llm-2/i);
    await switchBuildPanelTab(page, 'Config');
    const prompt2Input = page.getByRole('textbox', { name: /Prompt/i });
    await prompt2Input.fill('LLM-2 Content');

    // Wait for autosave
    await page.waitForTimeout(1500);

    // Go back to llm-1 - its config should NOT have changed
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/LLM-1 Original/);
    await expect(page.getByRole('textbox', { name: /Prompt/i })).not.toHaveValue(/LLM-2/);
  });

  test('deleting a node does not affect other node configs', async ({ page }) => {
    test.setTimeout(90000);
    // Add 3 blocks
    await addBlock(page, 'llm'); // llm-1
    await addBlock(page, 'if_else'); // if-1
    await addBlock(page, 'llm'); // llm-2

    // Configure all
    await configureLLMBlock(page, /llm-1/i, 'First LLM stays');
    await configureIfElseBlock(page, /if/i, '{{x}} == 1');
    await configureLLMBlock(page, /llm-2/i, 'Second LLM also stays');

    // Wait for autosave
    await page.waitForTimeout(2000);

    // Delete the if/else node
    await selectNode(page, /if/i);
    await page.keyboard.press('Delete');
    // Or use backspace
    await page.keyboard.press('Backspace');

    // Wait for deletion to save
    await page.waitForTimeout(1500);

    // Verify remaining nodes still have correct configs
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/First LLM stays/);

    await selectNode(page, /llm-2/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Second LLM also stays/);
  });

  test('modifying node config immediately after adding new node', async ({ page }) => {
    test.setTimeout(60000); // Multiple selectNode calls may trigger organize layout
    // Add first block and configure
    await addBlock(page, 'llm');
    await configureLLMBlock(page, /llm-1/i, 'First config');

    // Immediately add second block (before autosave completes)
    await addBlock(page, 'llm');

    // Immediately modify second block
    await configureLLMBlock(page, /llm-2/i, 'Second config');

    // Wait for all saves
    await page.waitForTimeout(2500);

    // Verify both configs are correct
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/First config/);

    await selectNode(page, /llm-2/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Second config/);
  });

  test('workflow with trigger and multiple block types preserves all configs', async ({ page }) => {
    test.setTimeout(120000);
    // Build a realistic workflow
    await addTrigger(page, 'webhook.generic');
    await addBlock(page, 'llm');      // llm-1
    await addBlock(page, 'if_else');  // if-1
    await addBlock(page, 'llm');      // llm-2
    await addBlock(page, 'for_loop'); // for-1

    await organizeLayout(page);

    // Configure each block
    await configureLLMBlock(page, /llm-1/i, 'Analyze input data');
    await configureIfElseBlock(page, /if/i, '{{llm-1.category}} == "batch"');
    await configureLLMBlock(page, /llm-2/i, 'Process single item');

    // For loop needs array_variable config
    await selectNode(page, /loop/i);
    await switchBuildPanelTab(page, 'Config');
    const arrayInput = page.getByRole('textbox', { name: /Array source/i });
    if (await arrayInput.isVisible({ timeout: 3000 }).catch(() => false)) {
      await arrayInput.fill('{{trigger.items}}');
    }

    // Wait for all autosaves
    await page.waitForTimeout(3000);

    // Verify trigger exists
    await expect(page.locator('.react-flow__node').filter({ hasText: /trigger/i })).toBeVisible();

    // Verify all block configs
    await selectNode(page, /llm-1/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Analyze input/);

    await selectNode(page, /llm-2/i);
    await switchBuildPanelTab(page, 'Config');
    await expect(page.getByRole('textbox', { name: /Prompt/i })).toHaveValue(/Process single/);
  });
});

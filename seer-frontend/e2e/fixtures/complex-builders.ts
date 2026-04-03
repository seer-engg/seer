/**
 * Complex Workflow Builders for E2E Tests
 *
 * Higher-level utilities for scaffolding multi-node workflows.
 * These helpers reduce boilerplate when testing complex workflow scenarios.
 */
import type { Page } from '@playwright/test';
import {
  addTrigger,
  addBlock,
  connectNodes,
  organizeLayout,
  configureLLMBlock,
  selectNode,
  switchBuildPanelTab,
} from './utils';

/**
 * Blueprint for building a workflow programmatically.
 */
export interface WorkflowBlueprint {
  /** Trigger type (e.g., 'webhook.generic', 'poll.gmail.email_received') */
  trigger?: string;
  /** Block types to add (in order) */
  blocks: string[];
  /** Connections as [sourceIdx, targetIdx, branch?] where -1 = trigger */
  connections: [number, number, string?][];
  /** Optional configs keyed by node index (0-based, trigger = -1) */
  configs?: Record<number, { prompt?: string; condition?: string; title?: string }>;
}

/**
 * Build a workflow from a blueprint.
 * Creates nodes, organizes layout, and connects them.
 *
 * @example
 * await buildWorkflowFromBlueprint(page, {
 *   trigger: 'webhook.generic',
 *   blocks: ['llm', 'if_else', 'llm', 'llm'],
 *   connections: [[-1, 0], [0, 1], [1, 2, 'true'], [1, 3, 'false']]
 * });
 */
export async function buildWorkflowFromBlueprint(
  page: Page,
  blueprint: WorkflowBlueprint
): Promise<void> {
  // Add trigger if specified
  if (blueprint.trigger) {
    await addTrigger(page, blueprint.trigger);
  }

  // Add all blocks
  for (const blockType of blueprint.blocks) {
    await addBlock(page, blockType);
  }

  // Organize layout to prevent overlapping nodes
  await organizeLayout(page);

  // Create connections
  for (const connection of blueprint.connections) {
    const [srcIdx, tgtIdx] = connection;
    // Note: branch (connection[2]) is available for future use when implementing
    // specific handle targeting for if/else nodes
    const srcLabel = getNodeLabel(srcIdx, blueprint);
    const tgtLabel = getNodeLabel(tgtIdx, blueprint);
    await connectNodes(page, srcLabel, tgtLabel);
  }

  // Wait for autosave to complete
  await page.waitForTimeout(1500);
}

/**
 * Get a node label pattern from its index in the blueprint.
 */
function getNodeLabel(index: number, blueprint: WorkflowBlueprint): RegExp {
  if (index === -1) {
    return /trigger/i;
  }

  const blockType = blueprint.blocks[index];
  const blockNumber = countBlocksBefore(index, blockType, blueprint.blocks) + 1;

  // Map block types to their expected labels
  const labelPatterns: Record<string, string> = {
    llm: `llm-${blockNumber}`,
    if_else: `if-${blockNumber}|if_else-${blockNumber}`,
    for_loop: `for-${blockNumber}|for_loop-${blockNumber}`,
    hitl: `hitl-${blockNumber}`,
    image_gen: `image_gen-${blockNumber}|image-${blockNumber}`,
    browser: `browser-${blockNumber}`,
    mcp: `mcp-${blockNumber}`,
  };

  const pattern = labelPatterns[blockType] || `${blockType}-${blockNumber}`;
  return new RegExp(pattern, 'i');
}

/**
 * Count how many blocks of the same type appear before this index.
 */
function countBlocksBefore(index: number, blockType: string, blocks: string[]): number {
  let count = 0;
  for (let i = 0; i < index; i++) {
    if (blocks[i] === blockType) {
      count++;
    }
  }
  return count;
}

/**
 * Predefined blueprints for common workflow patterns.
 */
export const BLUEPRINTS = {
  /** Simple: Webhook → LLM */
  webhookToLlm: {
    trigger: 'webhook.generic',
    blocks: ['llm'],
    connections: [[-1, 0]],
  } as WorkflowBlueprint,

  /** Conditional: Webhook → LLM → If/Else → LLM (true) / LLM (false) */
  conditionalBranching: {
    trigger: 'webhook.generic',
    blocks: ['llm', 'if_else', 'llm', 'llm'],
    connections: [
      [-1, 0],    // trigger → llm-1
      [0, 1],     // llm-1 → if
      [1, 2],     // if (true) → llm-2
      [1, 3],     // if (false) → llm-3
    ],
  } as WorkflowBlueprint,

  /** Loop: Webhook → For Loop → LLM (inside loop) */
  loopWithLlm: {
    trigger: 'webhook.generic',
    blocks: ['for_loop', 'llm'],
    connections: [
      [-1, 0],    // trigger → for_loop
      [0, 1],     // for_loop → llm (loop body)
    ],
  } as WorkflowBlueprint,

  /** HITL: Webhook → LLM → HITL → LLM */
  humanInTheLoop: {
    trigger: 'webhook.generic',
    blocks: ['llm', 'hitl', 'llm'],
    connections: [
      [-1, 0],    // trigger → llm-1
      [0, 1],     // llm-1 → hitl
      [1, 2],     // hitl → llm-2
    ],
  } as WorkflowBlueprint,

  /** Gmail: Gmail trigger → LLM */
  gmailToLlm: {
    trigger: 'poll.gmail.email_received',
    blocks: ['llm'],
    connections: [[-1, 0]],
  } as WorkflowBlueprint,

  /** Complex: Webhook → LLM → If/Else → (For Loop → LLM) / HITL */
  complex: {
    trigger: 'webhook.generic',
    blocks: ['llm', 'if_else', 'for_loop', 'llm', 'hitl'],
    connections: [
      [-1, 0],    // trigger → llm-1
      [0, 1],     // llm-1 → if
      [1, 2],     // if (true) → for_loop
      [2, 3],     // for_loop → llm-2
      [1, 4],     // if (false) → hitl
    ],
  } as WorkflowBlueprint,
};

/**
 * Configure multiple nodes in a workflow.
 * Useful for setting up complex test scenarios.
 *
 * @example
 * await configureMultipleNodes(page, {
 *   'llm-1': { prompt: 'Analyze this' },
 *   'if-1': { condition: '{{llm-1.output}} == "success"' },
 *   'llm-2': { prompt: 'Draft response' },
 * });
 */
export async function configureMultipleNodes(
  page: Page,
  configs: Record<string, { prompt?: string; condition?: string; title?: string }>
): Promise<void> {
  for (const [nodeLabel, config] of Object.entries(configs)) {
    if (config.prompt) {
      await configureLLMBlock(page, new RegExp(nodeLabel, 'i'), config.prompt);
    } else if (config.condition) {
      await configureIfElseBlock(page, new RegExp(nodeLabel, 'i'), config.condition);
    } else if (config.title) {
      await configureHITLBlock(page, new RegExp(nodeLabel, 'i'), config.title);
    }
  }

  // Wait for all autosaves
  await page.waitForTimeout(2000);
}

/**
 * Configure an If/Else block with a condition.
 * (Helper that may need to be added to utils.ts)
 */
async function configureIfElseBlock(
  page: Page,
  nodeLabel: string | RegExp,
  condition: string
): Promise<void> {
  await selectNode(page, nodeLabel);
  await switchBuildPanelTab(page, 'Config');

  const conditionField = page.getByRole('textbox', { name: /condition/i });
  if (await conditionField.isVisible({ timeout: 3000 }).catch(() => false)) {
    await conditionField.fill(condition);
  }

  await page.waitForTimeout(1000);
}

/**
 * Configure a HITL block with a title.
 */
async function configureHITLBlock(
  page: Page,
  nodeLabel: string | RegExp,
  title: string
): Promise<void> {
  await selectNode(page, nodeLabel);
  await switchBuildPanelTab(page, 'Config');

  const titleField = page.getByRole('textbox', { name: /title/i });
  if (await titleField.isVisible({ timeout: 3000 }).catch(() => false)) {
    await titleField.fill(title);
  }

  await page.waitForTimeout(1000);
}

/**
 * Verify that all nodes in a workflow have expected configurations.
 * Throws an error if any verification fails.
 */
export async function verifyNodeConfigs(
  page: Page,
  expectations: Record<string, { prompt?: RegExp; condition?: RegExp; title?: RegExp }>
): Promise<void> {
  for (const [nodeLabel, expected] of Object.entries(expectations)) {
    await selectNode(page, new RegExp(nodeLabel, 'i'));
    await switchBuildPanelTab(page, 'Config');

    if (expected.prompt) {
      const promptField = page.getByRole('textbox', { name: /prompt/i });
      const value = await promptField.inputValue();
      if (!expected.prompt.test(value)) {
        throw new Error(`Node ${nodeLabel}: expected prompt matching ${expected.prompt}, got "${value}"`);
      }
    }

    if (expected.condition) {
      const conditionField = page.getByRole('textbox', { name: /condition/i });
      const value = await conditionField.inputValue();
      if (!expected.condition.test(value)) {
        throw new Error(`Node ${nodeLabel}: expected condition matching ${expected.condition}, got "${value}"`);
      }
    }

    if (expected.title) {
      const titleField = page.getByRole('textbox', { name: /title/i });
      const value = await titleField.inputValue();
      if (!expected.title.test(value)) {
        throw new Error(`Node ${nodeLabel}: expected title matching ${expected.title}, got "${value}"`);
      }
    }
  }
}

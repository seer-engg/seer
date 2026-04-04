import type { FunctionBlockSchema, GmailIntegrationContext, DiscordIntegrationContext, SupabaseIntegrationContext, WorkflowNodeData } from '@/components/workflows/types';
import { GMAIL_TRIGGER_KEY, DISCORD_TRIGGER_KEY, SUPABASE_TRIGGER_KEY } from '@/components/workflows/triggers/constants';
import { getUserDefaultModel } from '@/lib/default-model';

/** Block type to node ID prefix mapping */
const BLOCK_TYPE_PREFIXES: Record<string, string> = {
  if_else: 'if',
  for_loop: 'loop',
  mcp: 'mcp',
  hitl: 'hitl',
  ea_bot: 'ea_bot',
  browser: 'browser',
  image_gen: 'img',
  task: 'task',
  agent: 'agent',
};

/**
 * Extract the base type and counter from a node ID
 * Examples: "agent-1" -> { prefix: "agent", counter: 1 }, "trigger-2" -> { prefix: "trigger", counter: 2 }
 */
function parseNodeId(nodeId: string): { prefix: string; counter: number } | null {
  const match = nodeId.match(/^(.+)-(\d+)$/);
  if (!match) return null;
  return {
    prefix: match[1],
    counter: parseInt(match[2], 10),
  };
}

/**
 * Get node ID prefix for a block type
 */
function getBlockPrefix(blockType: string, toolName?: string): string {
  if (blockType === 'tool') {
    return toolName ? toolName.toLowerCase().replace(/\s+/g, '_') : 'tool';
  }
  return BLOCK_TYPE_PREFIXES[blockType] ?? blockType;
}

/**
 * Generate unique node ID based on type and existing nodes
 * Creates counter-based IDs like "llm-1", "llm-2", "tool-1", "trigger-1"
 */
export function generateNodeId(
  type: 'trigger' | 'block',
  existingNodeIds: string[],
  context?: {
    blockType?: string;
    toolName?: string;
    model?: string;
    triggerKey?: string;
  }
): string {
  // Determine the prefix for the new node
  let prefix: string;

  if (type === 'trigger') {
    prefix = 'trigger';
  } else if (context?.blockType) {
    prefix = getBlockPrefix(context.blockType, context.toolName);
  } else {
    prefix = 'node';
  }

  // Find the highest counter for this prefix
  let maxCounter = 0;
  for (const nodeId of existingNodeIds) {
    const parsed = parseNodeId(nodeId);
    if (parsed && parsed.prefix === prefix) {
      maxCounter = Math.max(maxCounter, parsed.counter);
    }
  }

  // Return the next counter value
  return `${prefix}-${maxCounter + 1}`;
}

/**
 * Apply default configuration for a given block type
 */
export function withDefaultBlockConfig(
  blockType: string,
  config: Record<string, unknown> = {},
  functionBlockMap?: Map<string, FunctionBlockSchema>,
): Record<string, unknown> {
  const schemaDefaults = functionBlockMap?.get(blockType)?.defaults;
  // Only use schemaDefaults if it has actual properties (not empty object from backend)
  const hasSchemaDefaults = schemaDefaults && Object.keys(schemaDefaults).length > 0;
  const defaults: Record<string, unknown> = hasSchemaDefaults ? schemaDefaults : (() => {
    switch (blockType) {
      case 'if_else':
        return {
          condition: '',
        };
      case 'for_loop':
        return {
          array_mode: 'variable',
          array_variable: 'items',
          array_literal: [],
          item_var: 'item',
        };
      case 'mcp':
        return {
          server_type: 'http',
          server: '',
          tool: '',
          inputs: {},
        };
      case 'hitl':
        return {
          title: 'Approval Required',
          description: '',
          display: [],
          inputs: [
            {
              id: 'approval',
              question: 'Do you approve?',
              input_type: 'single_choice',
              options: [
                { value: 'approve', label: 'Approve' },
                { value: 'reject', label: 'Reject' },
              ],
              required: true,
            },
          ],
          timeout_seconds: 86400,
          output_schema: {
            type: 'object',
            properties: {
              approval: { type: 'string', enum: ['approve', 'reject'] },
            },
          },
          delivery_channels: [{ type: 'platform' }],
        };
      case 'ea_bot':
        return {
          meeting_url: '',
          bot_name: 'meetingsEA',
          recording_mode: 'speaker_view',
          timeout_seconds: 14400,
        };
      case 'browser':
        return {
          task: '',
          inputs: {},
          browser_profile_id: null,
          max_steps: 25,
          timeout_seconds: 300,
          save_screenshots: false,
        };
      case 'image_gen':
        return {
          prompt: '',
          model: 'sourceful/riverflow-v2-fast',
          size: '1024x1024',
        };
      case 'agent':
        return {
          prompt: '',
          model: getUserDefaultModel(),
          memory_bank_id: null,
          tools: ['web_search'],
          tool_connections: {},
          max_iterations: 10,
        };
      case 'trigger':
        // Triggers have their config passed directly
        return {};
      default:
        return {};
    }
  })();

  return {
    ...defaults,
    ...config,
  };
}

/**
 * Build integration metadata for trigger nodes
 */
export function buildIntegrationMetadata(
  triggerKey: string,
  gmailIntegration?: GmailIntegrationContext,
  discordIntegration?: DiscordIntegrationContext,
  supabaseIntegration?: SupabaseIntegrationContext
): NonNullable<WorkflowNodeData['triggerMeta']>['integration'] {
  const integrationMeta: NonNullable<WorkflowNodeData['triggerMeta']>['integration'] = {};

  if (triggerKey === GMAIL_TRIGGER_KEY && gmailIntegration) {
    integrationMeta.gmail = {
      ready: gmailIntegration.ready,
      connectionId: gmailIntegration.connectionId,
      onConnect: gmailIntegration.ready ? undefined : gmailIntegration.onConnect,
      isConnecting: gmailIntegration.isConnecting,
    };
  }

  if (triggerKey === DISCORD_TRIGGER_KEY && discordIntegration) {
    integrationMeta.discord = {
      ready: discordIntegration.ready,
      connectionId: discordIntegration.connectionId,
      onConnect: discordIntegration.ready ? undefined : discordIntegration.onConnect,
      isConnecting: discordIntegration.isConnecting,
    };
  }

  if (triggerKey === SUPABASE_TRIGGER_KEY && supabaseIntegration) {
    integrationMeta.supabase = {
      ready: supabaseIntegration.ready,
      onConnect: supabaseIntegration.ready ? undefined : supabaseIntegration.onConnect,
      isConnecting: supabaseIntegration.isConnecting,
    };
  }

  return Object.keys(integrationMeta).length ? integrationMeta : undefined;
}

/**
 * Create trigger node data structure
 */
export function createTriggerNodeData(options: {
  triggerKey: string;
  label: string;
  descriptor: { title?: string };
  workflowInputs: Record<string, unknown>;
  integrationMeta?: NonNullable<WorkflowNodeData['triggerMeta']>['integration'];
}): WorkflowNodeData {
  const { triggerKey, label, descriptor, workflowInputs, integrationMeta } = options;

  return {
    type: 'trigger',
    label: descriptor.title || label,
    config: {
      triggerKey,
    },
    triggerMeta: {
      descriptor,
      workflowInputs,
      handlers: {}, // Will be populated when trigger nodes are created
      integration: integrationMeta,
    },
  };
}

/**
 * Create regular block node data structure
 */
export function createBlockNodeData(options: {
  type: string;
  label: string;
  config: Record<string, unknown>;
}): WorkflowNodeData {
  const { type, label, config } = options;

  return {
    type: type as WorkflowNodeData['type'],
    label,
    config,
  };
}

import type { Node } from '@xyflow/react';
import type { WorkflowNodeData } from '@/components/workflows/types';
import type { TriggerCatalogEntry } from '@/types/triggers';
import type { BlockSelectionPayload } from '@/types/block-selection';
import { withDefaultBlockConfig as withDefaults, generateNodeId } from '@/lib/workflow-nodes';
import { GMAIL_TRIGGER_KEY, DISCORD_TRIGGER_KEY } from '@/components/workflows/triggers/constants';

export interface TriggerNodeCreationParams {
  id?: string;
  triggerKey: string;
  descriptor: TriggerCatalogEntry;
  // OAuth flow context (connectionId, onConnect, isConnecting still needed for connect button)
  gmailConnectionId: number | null;
  handleConnectGmail: () => void;
  isConnectingGmail: boolean;
  discordConnectionId: number | null;
  handleConnectDiscord: () => void;
  isConnectingDiscord: boolean;
  position: { x: number; y: number };
  existingNodeIds: string[];
}

export function createTriggerNode(params: TriggerNodeCreationParams): Node<WorkflowNodeData> {
  const {
    id,
    triggerKey,
    descriptor,
    gmailConnectionId,
    handleConnectGmail,
    isConnectingGmail,
    discordConnectionId,
    handleConnectDiscord,
    isConnectingDiscord,
    position,
    existingNodeIds,
  } = params;

  // Use backend-provided is_connected status (fallback to false for backwards compatibility)
  const isConnected = descriptor.is_connected ?? false;

  const integrationMeta: NonNullable<WorkflowNodeData['triggerMeta']>['integration'] = {};
  if (triggerKey === GMAIL_TRIGGER_KEY) {
    integrationMeta.gmail = {
      ready: isConnected,
      connectionId: gmailConnectionId,
      onConnect: isConnected ? undefined : handleConnectGmail,
      isConnecting: isConnectingGmail,
    };
  }

  if (triggerKey === DISCORD_TRIGGER_KEY) {
    integrationMeta.discord = {
      ready: isConnected,
      connectionId: discordConnectionId,
      onConnect: isConnected ? undefined : handleConnectDiscord,
      isConnecting: isConnectingDiscord,
    };
  }

  const triggerId = id || generateNodeId('trigger', existingNodeIds, { triggerKey });
  return {
    id: triggerId,
    type: 'trigger',
    position,
    data: {
      type: 'trigger',
      label: triggerId, // Use generated ID as label
      config: {
        triggerKey,
      },
      triggerMeta: {
        descriptor,
        handlers: {} as Record<string, (data: unknown) => void>,
        integration: Object.keys(integrationMeta).length ? integrationMeta : undefined,
      },
    },
  };
}

export function createRegularBlockNode(
  block: BlockSelectionPayload,
  position: { x: number; y: number },
  functionBlocksMap: Map<string, unknown>,
  existingNodeIds: string[],
): Node<WorkflowNodeData> {
  const defaultConfig = withDefaults(block.type, block.config, functionBlocksMap);

  // Extract context for descriptive ID generation
  const context: Parameters<typeof generateNodeId>[2] = {
    blockType: block.type,
  };

  // Add type-specific context
  if (block.type === 'tool' && block.config?.tool_name) {
    context.toolName = block.config.tool_name as string;
  }

  const nodeId = generateNodeId('block', existingNodeIds, context);
  return {
    id: nodeId,
    type: block.type as WorkflowNodeData['type'],
    position,
    data: {
      type: block.type as WorkflowNodeData['type'],
      label: nodeId, // Use generated ID as label
      config: defaultConfig,
    },
  };
}

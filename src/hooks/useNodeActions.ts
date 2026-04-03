import { useCallback, useMemo } from 'react';
import type { Node } from '@xyflow/react';
import type {
  WorkflowNodeData,
  WorkflowEdge,
  WorkflowNodeUpdateOptions,
  BlockSelectionPayload,
} from '@/components/workflows/types';
import { toast } from '@/components/ui/sonner';
import { useFunctionBlocksQuery } from '@/hooks/useFunctionBlocksQuery';
import { useToolingData } from '@/hooks/useToolingData';
import { useTriggerCatalogQuery } from '@/hooks/useTriggerCatalogQuery';
import { useActiveWorkflowId } from '@/hooks/useActiveWorkflowId';
import { useCanvasStore } from '@/stores';
import { useUIStore } from '@/stores/uiStore';
import { useTriggersStore } from '@/stores/triggersStore';
import { createTriggerNode, createRegularBlockNode } from '../components/workflows/canvas/nodeCreation';
import { useCanvasMutations } from './useCanvasMutations';
import type { TriggerDescriptor } from '@/types/triggers';
import type { JsonObject } from '@/types/workflow-spec';
import { generateNodeId } from '@/lib/workflow-nodes';
import { getOAuthProvider, type IntegrationType } from '@/lib/integrations/client';

function wouldNodeUpdateChangeData(
  node: Node<WorkflowNodeData>,
  updates: Partial<WorkflowNodeData>,
): boolean {
  const mergedData: WorkflowNodeData = {
    ...node.data,
    ...updates,
  };

  if (updates.config) {
    const originalConfig = node.data?.config || {};
    const mergedConfig = {
      ...originalConfig,
      ...updates.config,
    } as Record<string, unknown>;

    if ('params' in updates.config || 'params' in originalConfig) {
      const originalParams =
        originalConfig.params && typeof originalConfig.params === 'object'
          ? originalConfig.params as Record<string, unknown>
          : {};
      const updateParams =
        updates.config.params && typeof updates.config.params === 'object'
          ? updates.config.params as Record<string, unknown>
          : {};

      mergedConfig.params = {
        ...originalParams,
        ...updateParams,
      };
    }

    if ('fields' in updates.config) {
      mergedConfig.fields = updates.config.fields;
    }

    mergedData.config = mergedConfig;
  }

  return JSON.stringify(mergedData) !== JSON.stringify(node.data);
}

/**
 * Creates initial event schema for a new trigger based on its descriptor.
 * For webhook/form triggers, creates a default user schema in event.properties.data.
 * For other triggers, copies the event_schema from the descriptor.
 */
function createInitialEventSchema(descriptor: TriggerDescriptor): JsonObject {
  const isWebhookOrForm = descriptor.key === 'webhook.generic' || descriptor.key === 'form.hosted';

  if (isWebhookOrForm) {
    // Create default webhook envelope with empty data schema
    const defaultDataSchema: JsonObject = {
      type: 'object',
      properties: {},
      additionalProperties: false,
    };

    // If descriptor has an event_schema template, use it
    if (descriptor.event_schema) {
      const baseSchema = descriptor.event_schema as JsonObject;
      const properties = (baseSchema.properties as JsonObject) ?? {};

      return {
        ...baseSchema,
        properties: {
          ...properties,
          data: defaultDataSchema,
        },
      } as JsonObject;
    }

    // Default webhook envelope structure
    return {
      type: 'object',
      properties: {
        headers: {
          type: 'object',
          additionalProperties: { type: 'string' },
        },
        query: {
          type: 'object',
          additionalProperties: { type: 'string' },
        },
        data: defaultDataSchema,
      },
      required: ['data'],
    };
  }

  // For other triggers (Gmail, Cron, Supabase), copy event_schema from descriptor
  return descriptor.event_schema ?? {};
}

/**
 * Finds the provider connection ID for a trigger using multiple lookup strategies.
 * Handles cases where provider names don't match exactly (e.g., "supabase" vs "supabase_mgmt").
 * Also handles OAuth provider mapping (e.g., "gmail" integration → "google" OAuth provider).
 * Returns an integer ID extracted from the connectionId string (format: "provider:id" → id as number).
 */
function findProviderConnectionId(
  descriptor: { provider: string },
  triggerKey: string,
  providerConnectionsMap: Map<string, { connectionId: string; scopes: Set<string> }>
): number | null {
  let providerInfo: { connectionId: string; scopes: Set<string> } | undefined;
  let providerName = descriptor.provider;

  // Strategy 1: Try descriptor.provider directly
  providerInfo = providerConnectionsMap.get(providerName);

  // Strategy 2: Extract from trigger key (e.g., "webhook.supabase.db_changes" → "supabase")
  if (!providerInfo && triggerKey.includes('.')) {
    const parts = triggerKey.split('.');
    if (parts.length >= 2) {
      providerName = parts[1];
      providerInfo = providerConnectionsMap.get(providerName);
    }
  }

  // Strategy 3: Try OAuth provider mapping (e.g., "gmail" → "google")
  // This handles cases where integration type differs from OAuth provider name
  if (!providerInfo) {
    const oauthProvider = getOAuthProvider(providerName as IntegrationType);
    if (oauthProvider) {
      providerInfo = providerConnectionsMap.get(oauthProvider);
    }
  }

  // Strategy 4: Try with "_mgmt" suffix (e.g., "supabase" → "supabase_mgmt")
  if (!providerInfo) {
    const mgmtKey = `${providerName}_mgmt`;
    providerInfo = providerConnectionsMap.get(mgmtKey);
  }

  // Strategy 5: Partial match - find any key that starts with the provider name
  if (!providerInfo) {
    const matchingKey = Array.from(providerConnectionsMap.keys()).find(key =>
      key.startsWith(providerName) || providerName.startsWith(key)
    );
    if (matchingKey) {
      providerInfo = providerConnectionsMap.get(matchingKey);
    }
  }

  // Extract integer ID from connectionId string (format: "provider:id" → extract id as number)
  if (providerInfo?.connectionId) {
    const connectionIdStr = providerInfo.connectionId;
    const parts = connectionIdStr.split(':');
    if (parts.length === 2) {
      const id = parseInt(parts[1], 10);
      if (!isNaN(id)) {
        return id;
      }
    }
  }

  return null;
}

/**
 * Consolidated node actions hook that combines:
 * - useNodeHandlers (add nodes, double-click, dialog management)
 * - useNodeConfigUpdate (update node config with autosave)
 *
 * Phase 3 refactoring: Fetches all required state directly from stores
 * instead of receiving it through props.
 *
 * Connection status now comes from backend via trigger descriptor's is_connected field.
 */
export function useNodeActions({
  autosaveCallback,
  triggerSave,
  gmailConnectionId,
  handleConnectGmail,
  isConnectingGmail,
  discordConnectionId,
  handleConnectDiscord,
  isConnectingDiscord,
}: {
  autosaveCallback: (data: { nodes: Node<WorkflowNodeData>[]; edges: WorkflowEdge[] }) => Promise<void>;
  triggerSave: () => void;
  gmailConnectionId: number | null;
  handleConnectGmail: () => void;
  isConnectingGmail: boolean;
  discordConnectionId: number | null;
  handleConnectDiscord: () => void;
  isConnectingDiscord: boolean;
}) {
  // Unified mutation layer for all canvas operations
  const mutations = useCanvasMutations({
    triggerSave,
    saveImmediately: async () => {
      const { nodes, edges } = useCanvasStore.getState();
      await autosaveCallback({ nodes, edges });
    },
  });
  // Fetch required state from stores - FIXED: Individual selectors instead of useShallow
  const setSelectedNodeId = useCanvasStore((state) => state.setSelectedNodeId);
  const setRightPaneMode = useUIStore((state) => state.setRightPaneMode);

  const activeWorkflowId = useActiveWorkflowId();
  const { functionBlocksMap } = useFunctionBlocksQuery();
  const { providerConnectionsMap } = useToolingData();
  const { data: triggerCatalog = [] } = useTriggerCatalogQuery();

  // Get workflow inputs definition from current workflow
  const workflowInputsDef = useMemo(() => ({}), []);

  // ==================== NODE HANDLERS ====================

  /**
   * Add a block to the canvas (from BuildPanel)
   */
  const handleBlockSelect = useCallback(
    (block: BlockSelectionPayload) => {
      const position = { x: Math.random() * 400 + 100, y: Math.random() * 400 + 100 };
      const { nodes } = useCanvasStore.getState();
      const existingNodeIds = nodes.map((n) => n.id);

      if (block.type === 'trigger' && block.config?.triggerKey) {
        const triggerKey = block.config.triggerKey;
        const descriptor = triggerCatalog.find((t) => t.key === triggerKey);

        if (!descriptor) {
          toast.error('Trigger metadata unavailable');
          return;
        }

        // Generate unique trigger ID
        const triggerId = generateNodeId('trigger', existingNodeIds);

        const newNode = createTriggerNode({
          id: triggerId,
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
        });
        mutations.addNode(newNode);

        // Add trigger to spec with initialized event schema
        if (activeWorkflowId) {
          // Get connection ID for this provider from the connections map
          const providerConnectionId = findProviderConnectionId(descriptor, triggerKey, providerConnectionsMap);

          useTriggersStore.getState().addTrigger(
            activeWorkflowId,
            triggerId,
            {
              key: triggerKey,
              mode: descriptor.mode,
              event_schema: createInitialEventSchema(descriptor),
              meta: { requires_connection: descriptor.requires_connection ?? false },
              filters: {},
              provider_config: providerConnectionId ? { provider_connection_id: providerConnectionId } : {},
              ui_meta: {},
            }
          );
        }
        return;
      }

      const newNode = createRegularBlockNode(block, position, functionBlocksMap, existingNodeIds);
      mutations.addNode(newNode);
    },
    [
      triggerCatalog,
      gmailConnectionId,
      handleConnectGmail,
      isConnectingGmail,
      discordConnectionId,
      handleConnectDiscord,
      isConnectingDiscord,
      functionBlocksMap,
      mutations,
      activeWorkflowId,
      providerConnectionsMap,
    ],
  );

  /**
   * Handle node drop (from drag and drop)
   */
  const handleNodeDrop = useCallback(
    (block: BlockSelectionPayload, position: { x: number; y: number }) => {
      const { nodes } = useCanvasStore.getState();
      const existingNodeIds = nodes.map((n) => n.id);

      if (block.type === 'trigger' && block.config?.triggerKey) {
        const triggerKey = block.config.triggerKey;
        const descriptor = triggerCatalog.find((t) => t.key === triggerKey);

        if (!descriptor) {
          toast.error('Trigger metadata unavailable');
          return;
        }

        // Generate unique trigger ID
        const triggerId = generateNodeId('trigger', existingNodeIds);

        const newNode = createTriggerNode({
          id: triggerId,
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
        });
        mutations.addNode(newNode);

        // Add trigger to spec with initialized event schema
        if (activeWorkflowId) {
          // Get connection ID for this provider from the connections map
          const providerConnectionId = findProviderConnectionId(descriptor, triggerKey, providerConnectionsMap);

          useTriggersStore.getState().addTrigger(
            activeWorkflowId,
            triggerId,
            {
              key: triggerKey,
              mode: descriptor.mode,
              event_schema: createInitialEventSchema(descriptor),
              meta: { requires_connection: descriptor.requires_connection ?? false },
              filters: {},
              provider_config: providerConnectionId ? { provider_connection_id: providerConnectionId } : {},
              ui_meta: {},
            }
          );
        }
        return;
      }

      const newNode = createRegularBlockNode(block, position, functionBlocksMap, existingNodeIds);
      mutations.addNode(newNode);
    },
    [
      triggerCatalog,
      gmailConnectionId,
      handleConnectGmail,
      isConnectingGmail,
      discordConnectionId,
      handleConnectDiscord,
      isConnectingDiscord,
      functionBlocksMap,
      mutations,
      activeWorkflowId,
      providerConnectionsMap,
    ],
  );

  /**
   * Handle single-click on canvas node (selects node; config shown inline in Build panel)
   */
  const handleCanvasNodeClick = useCallback(
    (node: Node<WorkflowNodeData>) => {
      setSelectedNodeId(node.id);
      setRightPaneMode('config');
    },
    [setSelectedNodeId, setRightPaneMode],
  );

  // ==================== NODE CONFIG UPDATE ====================

  /**
   * Update node configuration with optional immediate persistence
   */
  const handleNodeConfigUpdate = useCallback(
    async (
      nodeId: string,
      updates: Partial<WorkflowNodeData>,
      options?: WorkflowNodeUpdateOptions,
    ) => {
      const currentNode = useCanvasStore.getState().nodes.find((node) => node.id === nodeId);
      if (currentNode && !wouldNodeUpdateChangeData(currentNode, updates)) {
        return;
      }

      // Use unified mutation layer with immediate save for config updates
      mutations.updateNodeConfig(nodeId, updates, {
        immediate: options?.persist || false,
      });
    },
    [mutations],
  );

  return {
    // Node handlers
    handleBlockSelect,
    handleNodeDrop,
    handleCanvasNodeClick,

    // Node config update
    handleNodeConfigUpdate,
  };
}

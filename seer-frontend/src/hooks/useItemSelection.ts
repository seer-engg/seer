import { useCallback } from 'react';
import type { UnifiedItem } from '@/components/workflows/buildtypes';
import { useTriggerCatalogQuery } from '@/hooks/useTriggerCatalogQuery';
import { useUIStore } from '@/stores/uiStore';
import { useCanvasStore } from '@/stores';
import { useTriggersStore } from '@/stores/triggersStore';
import { useWorkflowStore } from '@/stores/workflowStore';
import { withDefaultBlockConfig, generateNodeId } from '@/lib/workflow-nodes';
import { insertNodeBetweenEdge, createEdgeFromSource } from '@/lib/edge-insertion';
import type { Node } from '@xyflow/react';
import type { WorkflowNodeData } from '@/components/workflows/types';
import type { BlockSelectionPayload } from '@/types/block-selection';
import type { JsonObject } from '@/types/workflow-spec';
import type { TriggerDescriptor } from '@/types/triggers';
import { toast } from '@/components/ui/sonner';

/**
 * Calculate smart position for a new trigger node.
 * Places triggers above the workflow or to the right of existing triggers.
 */
function calculateTriggerPosition(nodes: Node<WorkflowNodeData>[]): { x: number; y: number } {
  const existingTriggers = nodes.filter((n) => n.data.type === 'trigger');

  if (existingTriggers.length === 0) {
    // No existing triggers - place above workflow or at default position
    const workflowNodes = nodes.filter((n) => n.data.type !== 'trigger');
    if (workflowNodes.length > 0) {
      const minY = Math.min(...workflowNodes.map((n) => n.position.y));
      const minX = Math.min(...workflowNodes.map((n) => n.position.x));
      return { x: minX, y: minY - 200 };
    }
    return { x: 200, y: 200 };
  }

  // Place to the right of the rightmost existing trigger
  const maxX = Math.max(...existingTriggers.map((n) => n.position.x));
  const referenceY = existingTriggers[0].position.y;
  return { x: maxX + 280, y: referenceY };
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
 * Convert UnifiedItem to BlockSelectionPayload
 */
function convertItemToBlockConfig(item: UnifiedItem): BlockSelectionPayload {
  if (item.type === 'block') {
    return { type: item.blockType, label: item.label };
  } else if (item.type === 'action') {
    // Handle integration groups - auto-select first tool
    if (item.integrationGroup) {
      const defaultTool = item.integrationGroup.tools[0];
      return {
        type: 'tool',
        label: defaultTool.name,
        config: {
          tool_name: defaultTool.name,
          provider: defaultTool.provider,
          integration_type: item.integrationGroup.integration_type,
          ...(defaultTool.output_schema ? { output_schema: defaultTool.output_schema } : {}),
          params: {},
        },
      };
    }
    // Legacy: handle individual tools
    if (item.tool) {
      return {
        type: 'tool',
        label: item.tool.name,
        config: {
          tool_name: item.tool.slug || item.tool.name,
          provider: item.tool.provider,
          integration_type: item.tool.integration_type,
          ...(item.tool.output_schema ? { output_schema: item.tool.output_schema } : {}),
          params: {},
        },
      };
    }
  } else if (item.type === 'trigger') {
    return {
      type: 'trigger',
      label: item.label,
      config: { triggerKey: item.triggerKey },
    };
  }
  // Fallback
  return { type: 'tool', label: 'Unknown' };
}

export function useItemSelection(
  onBlockSelect?: (block: { type: string; label: string; config?: Record<string, unknown> }) => void
) {
  const pendingConnection = useUIStore((state) => state.pendingConnection);
  const clearPendingConnection = useUIStore((state) => state.clearPendingConnection);
  const handleOrientation = useUIStore((state) => state.handleOrientation);
  const nodes = useCanvasStore((state) => state.nodes);
  const edges = useCanvasStore((state) => state.edges);
  const addNode = useCanvasStore((state) => state.addNode);
  const addEdge = useCanvasStore((state) => state.addEdge);
  const setNodes = useCanvasStore((state) => state.setNodes);
  const setEdges = useCanvasStore((state) => state.setEdges);
  const markDirty = useCanvasStore((state) => state.markDirty);

  // For trigger handling when no onBlockSelect callback is provided
  const { data: triggerCatalog = [] } = useTriggerCatalogQuery();
  const selectedWorkflowId = useWorkflowStore((state) => state.selectedWorkflowId);

  return useCallback(
    // eslint-disable-next-line complexity
    (item: UnifiedItem) => {
      const blockConfig = convertItemToBlockConfig(item);

      // Handle append mode - add node after source
      if (pendingConnection.mode === 'append' && pendingConnection.sourceNodeId) {
        // Special handling for triggers: place independently, don't connect to source
        if (blockConfig.type === 'trigger' && blockConfig.config?.triggerKey) {
          const triggerKey = blockConfig.config.triggerKey as string;
          const descriptor = triggerCatalog.find((t) => t.key === triggerKey);

          if (!descriptor) {
            toast.error('Trigger metadata unavailable');
            clearPendingConnection();
            return;
          }

          const existingNodeIds = nodes.map((n) => n.id);
          const position = calculateTriggerPosition(nodes);
          const triggerId = generateNodeId('trigger', existingNodeIds, { triggerKey });

          const newNode: Node<WorkflowNodeData> = {
            id: triggerId,
            type: 'trigger',
            position,
            data: {
              type: 'trigger',
              label: triggerId,
              config: { triggerKey },
              triggerMeta: {
                descriptor,
                handlers: {} as Record<string, (data: unknown) => void>,
              },
            },
          };
          addNode(newNode);

          // Add trigger to spec with initialized event schema
          if (selectedWorkflowId) {
            useTriggersStore.getState().addTrigger(
              selectedWorkflowId,
              triggerId,
              {
                key: triggerKey,
                mode: descriptor.mode,
                event_schema: createInitialEventSchema(descriptor),
                meta: { requires_connection: descriptor.requires_connection ?? false },
                filters: {},
                provider_config: {},
                ui_meta: {},
              }
            );
          }
          markDirty();
          clearPendingConnection();
          return;
        }

        const sourceNode = nodes.find((n) => n.id === pendingConnection.sourceNodeId);
        if (!sourceNode) {
          clearPendingConnection();
          return;
        }

        // Create new node positioned based on handle orientation
        const defaultConfig = withDefaultBlockConfig(
          blockConfig.type,
          blockConfig.config,
          new Map()
        );

        const existingNodeIds = nodes.map((n) => n.id);

        // Build context for descriptive ID generation
        const context: Parameters<typeof generateNodeId>[2] = {
          blockType: blockConfig.type,
        };

        // Add type-specific context
        if (blockConfig.type === 'tool' && blockConfig.config?.tool_name) {
          context.toolName = blockConfig.config.tool_name as string;
        } else if (blockConfig.type === 'llm' && defaultConfig.model) {
          context.model = defaultConfig.model as string;
        }

        const nodeId = generateNodeId('block', existingNodeIds, context);

        const newNode: Node<WorkflowNodeData> = {
          id: nodeId,
          type: blockConfig.type as WorkflowNodeData['type'],
          position:
            handleOrientation === 'vertical'
              ? { x: sourceNode.position.x, y: sourceNode.position.y + 200 }
              : { x: sourceNode.position.x + 300, y: sourceNode.position.y },
          data: {
            type: blockConfig.type as WorkflowNodeData['type'],
            label: nodeId, // Use generated ID as label
            config: defaultConfig,
          },
        };

        // Create edge from source to new node
        const newEdge = createEdgeFromSource(
          pendingConnection.sourceNodeId,
          newNode.id,
          pendingConnection.branch
        );

        addNode(newNode);
        addEdge(newEdge);
        markDirty();
        clearPendingConnection();
        return;
      }

      // Handle insert mode - insert node between existing edge
      if (pendingConnection.mode === 'insert' && pendingConnection.edgeId) {
        const edge = edges.find((e) => e.id === pendingConnection.edgeId);
        if (!edge) {
          clearPendingConnection();
          return;
        }

        const { nodes: updatedNodes, edges: updatedEdges } = insertNodeBetweenEdge(
          edge,
          blockConfig,
          nodes,
          edges,
          new Map()
        );

        setNodes(updatedNodes);
        setEdges(updatedEdges);
        markDirty();
        clearPendingConnection();
        return;
      }

      // Default behavior: pass to callback if provided, otherwise add node at canvas center
      if (onBlockSelect) {
        onBlockSelect(blockConfig);
        return;
      }

      // Empty canvas case — no source node, no callback: place node near canvas center
      const existingNodeIds = nodes.map((n) => n.id);
      const position = { x: 200, y: 200 };

      // Handle trigger blocks specially - need to set up triggerMeta and add to triggersStore
      if (blockConfig.type === 'trigger' && blockConfig.config?.triggerKey) {
        const triggerKey = blockConfig.config.triggerKey as string;
        const descriptor = triggerCatalog.find((t) => t.key === triggerKey);

        if (!descriptor) {
          toast.error('Trigger metadata unavailable');
          return;
        }

        const triggerId = generateNodeId('trigger', existingNodeIds, { triggerKey });
        const newNode: Node<WorkflowNodeData> = {
          id: triggerId,
          type: 'trigger',
          position,
          data: {
            type: 'trigger',
            label: triggerId,
            config: { triggerKey },
            triggerMeta: {
              descriptor,
              handlers: {} as Record<string, (data: unknown) => void>,
            },
          },
        };
        addNode(newNode);

        // Add trigger to spec with initialized event schema
        if (selectedWorkflowId) {
          useTriggersStore.getState().addTrigger(
            selectedWorkflowId,
            triggerId,
            {
              key: triggerKey,
              mode: descriptor.mode,
              event_schema: createInitialEventSchema(descriptor),
              meta: { requires_connection: descriptor.requires_connection ?? false },
              filters: {},
              provider_config: {},
              ui_meta: {},
            }
          );
        }
        markDirty();
        return;
      }

      // Regular block handling
      const defaultConfig = withDefaultBlockConfig(blockConfig.type, blockConfig.config, new Map());
      const context: Parameters<typeof generateNodeId>[2] = { blockType: blockConfig.type };
      if (blockConfig.type === 'tool' && blockConfig.config?.tool_name) {
        context.toolName = blockConfig.config.tool_name as string;
      }
      const nodeId = generateNodeId('block', existingNodeIds, context);
      const newNode: Node<WorkflowNodeData> = {
        id: nodeId,
        type: blockConfig.type as WorkflowNodeData['type'],
        position,
        data: {
          type: blockConfig.type as WorkflowNodeData['type'],
          label: nodeId,
          config: defaultConfig,
        },
      };
      addNode(newNode);
      markDirty();
    },
    [
      pendingConnection,
      handleOrientation,
      nodes,
      edges,
      addNode,
      addEdge,
      setNodes,
      setEdges,
      markDirty,
      clearPendingConnection,
      onBlockSelect,
      triggerCatalog,
      selectedWorkflowId,
    ]
  );
}

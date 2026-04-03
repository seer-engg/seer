/**
 * Unified Canvas Mutations Hook
 *
 * Central mutation layer that:
 * 1. Intercepts all canvas mutations (nodes, edges, config updates)
 * 2. Marks canvas as dirty
 * 3. Triggers unified debounced save
 * 4. Eliminates scattered save triggers and coordination hacks
 *
 * This replaces the previous scattered approach where save triggers
 * were spread across multiple hooks and components.
 */

import { useCallback, useRef } from 'react';
import type { Node } from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge } from '@/components/workflows/types';
import { useActiveWorkflowId } from '@/hooks/useActiveWorkflowId';
import { useCanvasStore } from '@/stores';
import { useTriggersStore } from '@/stores/triggersStore';

export interface MutationOptions {
  /**
   * If true, flushes the debounce and saves immediately.
   * Used for critical changes like config updates.
   */
  immediate?: boolean;

  /**
   * If true, skips marking as dirty and doesn't trigger save.
   * Used for non-persisted changes like selection or temporary states.
   */
  skipSave?: boolean;
}

export interface UseCanvasMutationsParams {
  /**
   * Debounced save trigger function from useDebouncedAutosave
   */
  triggerSave: () => void;

  /**
   * Immediate save function that bypasses debounce
   */
  saveImmediately?: () => Promise<void>;
}

/**
 * Unified mutation layer for all canvas operations.
 *
 * All mutations go through this hook, which ensures:
 * - Consistent save triggering
 * - Dirty state tracking
 * - Configurable save behavior per mutation type
 */
export function useCanvasMutations({ triggerSave, saveImmediately }: UseCanvasMutationsParams) {
  // NOTE: We intentionally do NOT subscribe to the store here.
  // Using useCanvasStore() would subscribe to ALL store fields, causing
  // re-renders on every canvas change. Instead, we use getState() inside
  // callbacks for write-only operations.
  const activeWorkflowId = useActiveWorkflowId();
  const pendingImmediateSaveRef = useRef(false);

  /**
   * Wrapper for all mutations that handles save orchestration
   */
  const mutate = useCallback(
    <T>(mutation: () => T, options?: MutationOptions): T => {
      // Execute the mutation
      const result = mutation();

      // Skip save logic if requested (e.g., for UI-only changes)
      if (options?.skipSave) {
        return result;
      }

      // Mark canvas as dirty (using getState() to avoid subscription)
      useCanvasStore.getState().markDirty();

      // Trigger save based on options
      if (options?.immediate && saveImmediately) {
        // Immediate save (for critical changes like config updates)
        // Debounce multiple immediate calls to avoid race conditions
        if (!pendingImmediateSaveRef.current) {
          pendingImmediateSaveRef.current = true;
          saveImmediately().finally(() => {
            pendingImmediateSaveRef.current = false;
          });
        }
      } else {
        // Debounced save (default for most operations)
        triggerSave();
      }

      return result;
    },
    [triggerSave, saveImmediately],
  );

  // ==================== Node Mutations ====================

  const addNode = useCallback(
    (node: Node<WorkflowNodeData>, options?: MutationOptions) => {
      return mutate(() => {
        useCanvasStore.getState().setNodes((nodes) => [...nodes, node]);
      }, options);
    },
    [mutate],
  );

  const removeNode = useCallback(
    (nodeId: string, options?: MutationOptions) => {
      return mutate(() => {
        const canvasState = useCanvasStore.getState();
        // Check if this is a trigger node before deleting
        const node = canvasState.nodes.find((n) => n.id === nodeId);
        if (node?.data?.type === 'trigger') {
          if (nodeId && activeWorkflowId) {
            // Use node ID (which equals trigger.id) to uniquely identify the trigger
            useTriggersStore.getState().removeTrigger(activeWorkflowId, nodeId);
          } else {
            // Log warning if trigger node is missing required metadata
            console.warn('Trigger node missing nodeId or workflowId', {
              nodeId,
              workflowId: activeWorkflowId,
            });
          }
        }
        canvasState.deleteNode(nodeId);
      }, options);
    },
    [activeWorkflowId, mutate],
  );

  const updateNodeConfig = useCallback(
    (nodeId: string, updates: Partial<WorkflowNodeData>, options?: MutationOptions) => {
      // Config updates default to immediate save
      const finalOptions: MutationOptions = {
        immediate: true,
        ...options,
      };

      return mutate(() => {
        useCanvasStore.getState().updateNode(nodeId, updates);
      }, finalOptions);
    },
    [mutate],
  );

  const updateNodePosition = useCallback(
    (nodeId: string, position: { x: number; y: number }, options?: MutationOptions) => {
      // Position updates don't save by default (too frequent)
      const finalOptions: MutationOptions = {
        skipSave: true,
        ...options,
      };

      const updatePosition = (nodes: Node<WorkflowNodeData>[]) =>
        nodes.map((node) => (node.id === nodeId ? { ...node, position } : node));

      return mutate(() => {
        useCanvasStore.getState().setNodes(updatePosition);
      }, finalOptions);
    },
    [mutate],
  );

  const setNodes = useCallback(
    (updater: Node<WorkflowNodeData>[] | ((prev: Node<WorkflowNodeData>[]) => Node<WorkflowNodeData>[]), options?: MutationOptions) => {
      return mutate(() => {
        useCanvasStore.getState().setNodes(updater);
      }, options);
    },
    [mutate],
  );

  // ==================== Edge Mutations ====================

  const addEdge = useCallback(
    (edge: WorkflowEdge, options?: MutationOptions) => {
      return mutate(() => {
        useCanvasStore.getState().addEdge(edge);
      }, options);
    },
    [mutate],
  );

  const removeEdge = useCallback(
    (edgeId: string, options?: MutationOptions) => {
      return mutate(() => {
        useCanvasStore.getState().deleteEdge(edgeId);
      }, options);
    },
    [mutate],
  );

  const setEdges = useCallback(
    (updater: WorkflowEdge[] | ((prev: WorkflowEdge[]) => WorkflowEdge[]), options?: MutationOptions) => {
      return mutate(() => {
        useCanvasStore.getState().setEdges(updater);
      }, options);
    },
    [mutate],
  );

  return {
    // Node mutations
    addNode,
    removeNode,
    updateNodeConfig,
    updateNodePosition,
    setNodes,

    // Edge mutations
    addEdge,
    removeEdge,
    setEdges,

    // Direct access to mutate for custom operations
    mutate,
  };
}

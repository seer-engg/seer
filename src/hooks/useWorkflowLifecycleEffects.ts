import { useEffect } from 'react';
import { toast } from '@/components/ui/sonner';
import type { Node } from '@xyflow/react';
import type { WorkflowNodeData } from '@/components/workflows/types';

interface UseWorkflowLifecycleEffectsProps {
  autosaveStatus: string;
  nodes: Node<WorkflowNodeData>[];
  workflowId: string | null;
  isLoadingWorkflow: boolean;
  isPreviewActive: boolean;
  isDirty: boolean;
  triggerSave: () => void;
  setNodes: (nodes: Node<WorkflowNodeData>[]) => void;
  setSelectedNodeId: (id: string | null) => void;
  triggerHandlers: unknown;
}

export function useWorkflowLifecycleEffects(props: UseWorkflowLifecycleEffectsProps) {
  const {
    autosaveStatus,
    nodes,
    workflowId,
    isLoadingWorkflow,
    isPreviewActive,
    isDirty,
    triggerSave,
    setNodes,
    setSelectedNodeId,
    triggerHandlers,
  } = props;

  // Trigger autosave when canvas becomes dirty
  // This replaces the previous count-based detection approach
  useEffect(() => {
    if (!workflowId || isLoadingWorkflow || !isDirty) {
      return;
    }
    triggerSave();
  }, [isDirty, workflowId, triggerSave, isLoadingWorkflow]);

  // Handle autosave errors
  useEffect(() => {
    if (autosaveStatus === 'error') {
      toast.error('Autosave failed', {
        description: 'Your changes may not have been saved. Please save manually.',
        duration: 5000,
      });
    }
  }, [autosaveStatus]);

  // Populate handlers for trigger nodes
  useEffect(() => {
    const hasUnpopulatedTriggerNodes = nodes.some(
      (node) =>
        node.type === 'trigger' &&
        node.data.triggerMeta &&
        (!node.data.triggerMeta.handlers || Object.keys(node.data.triggerMeta.handlers).length === 0),
    );

    if (hasUnpopulatedTriggerNodes) {
      const updatedNodes = nodes.map((node) => {
        if (
          node.type === 'trigger' &&
          node.data.triggerMeta &&
          (!node.data.triggerMeta.handlers || Object.keys(node.data.triggerMeta.handlers).length === 0)
        ) {
          return {
            ...node,
            data: {
              ...node.data,
              triggerMeta: {
                ...node.data.triggerMeta,
                handlers: triggerHandlers,
              },
            },
          };
        }
        return node;
      });
      setNodes(updatedNodes);
    }
  }, [nodes, triggerHandlers, setNodes]);

  // Clear selection when preview is active
  useEffect(() => {
    if (isPreviewActive) {
      setSelectedNodeId(null);
    }
  }, [isPreviewActive, setSelectedNodeId]);
}

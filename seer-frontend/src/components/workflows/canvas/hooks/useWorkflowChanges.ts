import { useCallback } from 'react';
import {
  Node,
  NodeChange,
  EdgeChange,
  applyNodeChanges,
  applyEdgeChanges,
} from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge } from '../../types';
import { useCanvasStore } from '@/stores/canvasStore';
import { useTriggersStore } from '@/stores/triggersStore';
import { useWorkflowStore } from '@/stores/workflowStore';

export function useWorkflowChanges(readOnly: boolean, selectedNodeId: string | null) {
  const setNodes = useCanvasStore((state) => state.setNodes);
  const setEdges = useCanvasStore((state) => state.setEdges);

  const handleNodesChange = useCallback(
    (changes: NodeChange<Node<WorkflowNodeData>>[]) => {
      if (readOnly) return;

      // Check if changes include mutations (not just selection/position)
      const hasMutation = changes.some(
        (change) => change.type === 'remove' || change.type === 'add' || change.type === 'replace',
      );

      // Handle trigger cleanup for removed nodes (keyboard deletion)
      const removeChanges = changes.filter((change) => change.type === 'remove');
      if (removeChanges.length > 0) {
        const currentNodes = useCanvasStore.getState().nodes;
        const workflowId = useWorkflowStore.getState().selectedWorkflowId;

        removeChanges.forEach((change) => {
          if (change.type === 'remove') {
            const node = currentNodes.find((n) => n.id === change.id);
            if (node?.data?.type === 'trigger' && workflowId) {
              // Remove trigger from store using node ID (which equals trigger.id)
              useTriggersStore.getState().removeTrigger(workflowId, node.id);
            }
          }
        });
      }

      // PHASE 3 FIX: Don't depend on renderedNodes - use functional update instead
      // This prevents infinite loops when renderedNodes changes after setNodes
      setNodes((currentNodes) => {
        // Apply selection state to current nodes before applying changes
        const nodesWithSelection = currentNodes.map((node) => {
          const isSelected = selectedNodeId ? node.id === selectedNodeId : false;
          return {
            ...node,
            data: {
              ...node.data,
              selected: isSelected,
            },
          };
        });
        return applyNodeChanges<Node<WorkflowNodeData>>(changes, nodesWithSelection);
      });

      // Mark canvas as dirty if there are actual mutations (unified save will handle the rest)
      if (hasMutation) {
        useCanvasStore.getState().markDirty();
      }
    },
    [readOnly, setNodes, selectedNodeId],
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange<WorkflowEdge>[]) => {
      if (readOnly) return;

      // Check if changes include mutations (not just selection)
      const hasMutation = changes.some(
        (change) => change.type === 'remove' || change.type === 'add',
      );

      setEdges((currentEdges) => applyEdgeChanges<WorkflowEdge>(changes, currentEdges));

      // Mark canvas as dirty if there are actual mutations
      if (hasMutation) {
        useCanvasStore.getState().markDirty();
      }
    },
    [readOnly, setEdges],
  );

  return { handleNodesChange, handleEdgesChange };
}

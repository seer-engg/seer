import { useMemo } from 'react';
import { Node } from '@xyflow/react';
import type { WorkflowNodeData } from '../../types';
import { useCanvasStore } from '@/stores';

export function useRenderedNodes(
  workflowNodes: Node<WorkflowNodeData>[],
  selectedNodeId: string | null,
): Node<WorkflowNodeData>[] {
  const nodeErrors = useCanvasStore(
    (state) => state.nodeErrors,
    (a, b) => {
      if (a.size !== b.size) return false;
      for (const [k, v] of a) {
        if (b.get(k) !== v) return false;
      }
      return true;
    },
  );

  return useMemo(() => {
    return workflowNodes.map((node) => {
      const isSelected = selectedNodeId ? node.id === selectedNodeId : false;
      const error = nodeErrors.get(node.id) ?? null;
      // Skip creating a new reference if nothing changed — React Flow
      // re-renders a node only when its object reference changes.
      if (node.data.selected === isSelected && node.data.error === error) {
        return node;
      }
      return {
        ...node,
        data: {
          ...node.data,
          selected: isSelected,
          error,
        },
      };
    });
  }, [workflowNodes, selectedNodeId, nodeErrors]);
}

import { useCallback } from 'react';
import { Node, NodeMouseHandler } from '@xyflow/react';
import type { WorkflowNodeData } from '../../types';

export function useNodeHandlers(
  readOnly: boolean,
  setSelectedNodeId: (id: string) => void,
  onNodeClick?: (node: Node<WorkflowNodeData>) => void,
) {
  const handleNodeClick: NodeMouseHandler = useCallback(
    (_, node) => {
      // Allow selection even in readOnly mode (e.g., for proposal preview)
      // so users can view node parameters in the config panel
      setSelectedNodeId(node.id);
      onNodeClick?.(node as Node<WorkflowNodeData>);
    },
    [setSelectedNodeId, onNodeClick],
  );

  return { handleNodeClick };
}

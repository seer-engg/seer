import { useCallback } from 'react';
import { addEdge, MarkerType, type Connection } from '@xyflow/react';
import type { Node } from '@xyflow/react';
import type { WorkflowEdge, WorkflowNodeData } from '../components/workflows/types';
import { getNextBranchForSource } from '../components/workflows/types';
import { useCanvasStore } from '@/stores';

type SetEdges = (
  edges:
    | WorkflowEdge[]
    | ((edges: WorkflowEdge[]) => WorkflowEdge[]),
) => void;

interface UseConnectionValidationParams {
  readOnly?: boolean;
  workflowNodes: Node<WorkflowNodeData>[];
  workflowEdges: WorkflowEdge[];
  setEdges: SetEdges;
}

const isTriggerConnectionInvalid = (
  sourceNode?: Node<WorkflowNodeData>,
  targetNode?: Node<WorkflowNodeData>,
) => {
  if (sourceNode?.type === 'trigger' && targetNode?.type === 'trigger') {
    console.warn('Cannot connect trigger to trigger');
    return true;
  }

  if (targetNode?.type === 'trigger') {
    console.warn('Cannot connect to trigger nodes (they are entry points)');
    return true;
  }

  return false;
};

const connectionCreatesCycle = (params: Connection, workflowEdges: WorkflowEdge[]) => {
  return workflowEdges.some((edge) => edge.target === params.source && edge.source === params.target);
};

const resolveBranch = (
  params: Connection,
  workflowNodes: Node<WorkflowNodeData>[],
  workflowEdges: WorkflowEdge[],
) => {
  const sourceNode = workflowNodes.find((node) => node.id === params.source);
  const branchFromHandle =
    params.sourceHandle && ['true', 'false', 'loop', 'exit'].includes(params.sourceHandle)
      ? (params.sourceHandle as 'true' | 'false' | 'loop' | 'exit')
      : undefined;

  if (sourceNode?.type === 'trigger') {
    return { branch: undefined, isTriggerEdge: true, valid: true, sourceNode };
  }

  const branch =
    branchFromHandle ?? getNextBranchForSource(params.source!, workflowNodes, workflowEdges);
  const needsBranchCheck = sourceNode && (sourceNode.type === 'if_else' || sourceNode.type === 'for_loop');

  if (!branch && needsBranchCheck) {
    console.warn(`All branch handles are already used for node ${sourceNode.id}`);
    return { branch: undefined, isTriggerEdge: false, valid: false, sourceNode };
  }

  return { branch, isTriggerEdge: false, valid: true, sourceNode };
};

const buildEdge = (
  params: Connection,
  branch: 'true' | 'false' | 'loop' | 'exit' | undefined,
  isTriggerEdge: boolean,
): WorkflowEdge => ({
  id: `edge-${params.source}-${params.target}`,
  source: params.source!,
  target: params.target!,
  // IMPORTANT: sourceHandle must match branch for conditional/loop edges
  // This ensures edges render from the correct visual handle position
  sourceHandle: branch ?? params.sourceHandle,
  targetHandle: params.targetHandle,
  data: branch ? { branch } : undefined,
  markerEnd: { type: MarkerType.ArrowClosed },
  ...(isTriggerEdge && {
    style: {
      stroke: 'hsl(239 84% 67%)',
      strokeWidth: 3,
      strokeDasharray: '5,5',
    },
  }),
});

export function useConnectionValidation({
  readOnly = false,
  workflowNodes,
  workflowEdges,
  setEdges,
}: UseConnectionValidationParams) {
  const onConnect = useCallback(
    (params: Connection) => {
      if (readOnly) return;
      if (!params.source || !params.target) return;

      const sourceNode = workflowNodes.find((node) => node.id === params.source);
      const targetNode = workflowNodes.find((node) => node.id === params.target);

      if (isTriggerConnectionInvalid(sourceNode, targetNode)) {
        return;
      }

      if (connectionCreatesCycle(params, workflowEdges)) {
        console.warn('Connection would create a cycle');
        return;
      }

      const { branch, isTriggerEdge, valid } = resolveBranch(params, workflowNodes, workflowEdges);
      if (!valid) {
        return;
      }

      setEdges((eds) => addEdge(buildEdge(params, branch, isTriggerEdge), eds));
      useCanvasStore.getState().markDirty();
    },
    [readOnly, workflowEdges, workflowNodes, setEdges],
  );

  return { onConnect } as const;
}

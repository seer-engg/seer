import { useMemo, useCallback } from 'react';
import { BUILT_IN_BLOCKS, getBlockIconForType } from '@/components/workflows/constants';
import type { FunctionBlock } from '@/types/workflow-spec';
import type { Node } from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge } from '@/components/workflows/types';
import { normalizeEdges, normalizeNodes } from '@/lib/workflow-normalization';

export function useWorkflowBlocks(params: {
  functionBlockSchemas: FunctionBlock[];
  functionBlocksMap: Map<string, FunctionBlock>;
  setNodes: (nodes: Node<WorkflowNodeData>[]) => void;
  setEdges: (edges: WorkflowEdge[]) => void;
  setProposalPreview: (preview: unknown) => void;
  setLastRunVersionId: (id: number | null) => void;
}) {
  const { functionBlockSchemas, functionBlocksMap, setNodes, setEdges, setProposalPreview, setLastRunVersionId } =
    params;
  const availableBlocks = useMemo(() => {
    const builtInBlocksMap = new Map(BUILT_IN_BLOCKS.map((block) => [block.type, block]));
    const mergedBlocks = [...BUILT_IN_BLOCKS];

    if (functionBlockSchemas.length > 0) {
      functionBlockSchemas.forEach((schema) => {
        if (!builtInBlocksMap.has(schema.type)) {
          mergedBlocks.push({
            type: schema.type,
            label: schema.label,
            description: schema.description,
            icon: getBlockIconForType(schema.type),
          });
        }
      });
    }

    return mergedBlocks;
  }, [functionBlockSchemas]);

  const handleWorkflowGraphSync = useCallback(
    (graph?: { nodes?: Node<WorkflowNodeData>[]; edges?: WorkflowEdge[] }) => {
      if (!graph) return;
      setNodes(normalizeNodes(graph.nodes, functionBlocksMap));
      setEdges(normalizeEdges(graph.edges));
      setProposalPreview(null);
      setLastRunVersionId(null);
    },
    [functionBlocksMap, setNodes, setEdges, setProposalPreview, setLastRunVersionId],
  );

  return { availableBlocks, handleWorkflowGraphSync };
}

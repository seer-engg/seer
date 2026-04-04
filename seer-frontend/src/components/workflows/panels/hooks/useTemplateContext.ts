import { useMemo } from 'react';
import { Node } from '@xyflow/react';
import { WorkflowNodeData, WorkflowEdge } from '../../types';
import { collectAvailableVariables, useTemplateAutocomplete } from '../../block-config';
import type { InputDef, TriggerSpec } from '@/types/workflow-spec';

interface UseTemplateContextParams {
  allNodes: Node<WorkflowNodeData>[];
  allEdges: WorkflowEdge[];
  node: Node<WorkflowNodeData> | null;
  workflowInputs?: Record<string, InputDef>;
  triggers?: TriggerSpec[];
}

export interface UseTemplateContextResult {
  availableVariables: ReturnType<typeof collectAvailableVariables>;
  templateAutocomplete: ReturnType<typeof useTemplateAutocomplete>;
}

export function useTemplateContext({
  allNodes,
  allEdges,
  node,
  workflowInputs,
  triggers,
}: UseTemplateContextParams): UseTemplateContextResult {
  const availableVariables = useMemo(
    () => collectAvailableVariables(allNodes, allEdges, node, workflowInputs, triggers),
    [allNodes, allEdges, node, workflowInputs, triggers],
  );

  const templateAutocomplete = useTemplateAutocomplete(availableVariables);

  return {
    availableVariables,
    templateAutocomplete,
  };
}

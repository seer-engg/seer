import { Node } from '@xyflow/react';
import { WorkflowNodeData } from '../../types';
import { ToolMetadata } from '../../block-config';
import { useToolCatalogQuery } from '@/hooks/useToolCatalogQuery';

interface UseToolSchemaParams {
  toolName: string;
  node: Node<WorkflowNodeData> | null;
}

export function useToolSchema({ toolName, node }: UseToolSchemaParams): ToolMetadata | undefined {
  const { data: tools = [] } = useToolCatalogQuery();

  if (!toolName || node?.data.type !== 'tool') {
    return undefined;
  }

  return tools.find((tool) => tool.name === toolName);
}

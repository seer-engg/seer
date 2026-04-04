import { memo } from 'react';
import { Globe } from 'lucide-react';
import { NodeProps } from '@xyflow/react';
import { WorkflowNodeData } from '../types';
import { BaseBlockNode } from './BaseBlockNode';

export const MCPBlockNode = memo(function MCPBlockNode(
  props: NodeProps<WorkflowNodeData>
) {
  const toolName = props.data?.config?.tool as string | undefined;

  return (
    <BaseBlockNode
      {...props}
      icon={<Globe className="w-4 h-4 text-cyan-500" />}
      color="cyan"
      handles={{
        inputs: ['input'],
        outputs: ['output'],
      }}
    >
      {toolName && (
        <p className="text-xs text-muted-foreground truncate">{toolName}</p>
      )}
    </BaseBlockNode>
  );
});

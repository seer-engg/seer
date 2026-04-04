import { memo, useMemo } from 'react';
import { Bot } from 'lucide-react';
import { NodeProps } from '@xyflow/react';
import { WorkflowNodeData } from '../types';
import { BaseBlockNode } from './BaseBlockNode';

export const AgentBlockNode = memo(function AgentBlockNode(
  props: NodeProps<WorkflowNodeData>
) {
  // Dynamically generate output handles based on structured output schema
  const outputHandles = useMemo(() => {
    const handles = ['output'];
    const outputSchema = props.data?.config?.output_schema;
    if (outputSchema && typeof outputSchema === 'object' && outputSchema.properties) {
      Object.keys(outputSchema.properties as Record<string, unknown>).forEach((fieldName) => {
        if (!handles.includes(fieldName)) {
          handles.push(fieldName);
        }
      });
    }
    return handles;
  }, [props.data?.config?.output_schema]);

  return (
    <BaseBlockNode
      {...props}
      icon={<Bot className="w-4 h-4 text-indigo-500" />}
      color="indigo"
      consumesCredits
      handles={{
        inputs: ['input'],
        outputs: outputHandles,
      }}
    />
  );
});

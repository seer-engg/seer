import { memo, useMemo } from 'react';
import { Monitor } from 'lucide-react';
import { NodeProps, type Node } from '@xyflow/react';
import { WorkflowNodeData } from '../types';
import { BaseBlockNode } from './BaseBlockNode';

type CanvasNode = Node<WorkflowNodeData>;

export const BrowserBlockNode = memo(function BrowserBlockNode(
  props: NodeProps<CanvasNode>
) {
  const task = props.data?.config?.task as string | undefined;
  const truncatedTask = task && task.length > 40 ? `${task.slice(0, 40)}...` : task;

  // Dynamically generate output handles based on structured output schema
  // Browser nodes always output wrapper fields: success, result, final_url, screenshots
  // Custom schema fields are nested under extracted_data
  const outputHandles = useMemo(() => {
    const handles = ['output']; // Always have the default output

    // Add standard browser output fields (always present from backend)
    handles.push('success', 'result', 'final_url', 'screenshots');

    const outputSchema = props.data?.config?.output_schema;
    if (outputSchema && typeof outputSchema === 'object' && 'properties' in outputSchema) {
      // Add extracted_data as a field
      handles.push('extracted_data');

      // Add nested handles for custom fields under extracted_data
      const properties = outputSchema.properties as Record<string, unknown>;
      Object.keys(properties).forEach((fieldName) => {
        const nestedHandle = `extracted_data.${fieldName}`;
        if (!handles.includes(nestedHandle)) {
          handles.push(nestedHandle);
        }
      });
    }

    return handles;
  }, [props.data?.config?.output_schema]);

  return (
    <BaseBlockNode
      {...props}
      icon={<Monitor className="w-4 h-4 text-violet-500" />}
      color="violet"
      consumesCredits
      handles={{
        inputs: ['input'],
        outputs: outputHandles,
      }}
    >
      {truncatedTask && (
        <p className="text-xs text-muted-foreground truncate">{truncatedTask}</p>
      )}
    </BaseBlockNode>
  );
});

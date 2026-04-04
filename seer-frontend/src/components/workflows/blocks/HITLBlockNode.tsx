import { memo } from 'react';
import { UserCheck } from 'lucide-react';
import { NodeProps } from '@xyflow/react';
import { WorkflowNodeData } from '../types';
import { BaseBlockNode } from './BaseBlockNode';

/**
 * HITL (Human-in-the-Loop) Block Node
 *
 * Displays a node that pauses workflow execution for human review and input.
 * Uses amber color to indicate the "waiting/attention" state.
 */
export const HITLBlockNode = memo(function HITLBlockNode(
  props: NodeProps<WorkflowNodeData>
) {
  return (
    <BaseBlockNode
      {...props}
      icon={<UserCheck className="w-4 h-4 text-amber-500" />}
      color="amber"
      handles={{
        inputs: ['input'],
        outputs: ['output'],
      }}
    />
  );
});

import { memo } from 'react';
import { Video } from 'lucide-react';
import { NodeProps } from '@xyflow/react';
import { WorkflowNodeData } from '../types';
import { BaseBlockNode } from './BaseBlockNode';

/**
 * meetingsEA Block Node
 *
 * Joins a meeting, records audio/video, and waits for the meeting to end.
 * Outputs transcript and recording URL when the meeting completes.
 */
export const EABotBlockNode = memo(function EABotBlockNode(
  props: NodeProps<WorkflowNodeData>
) {
  return (
    <BaseBlockNode
      {...props}
      icon={<Video className="w-4 h-4 text-violet-500" />}
      color="violet"
      handles={{
        inputs: ['input'],
        outputs: ['output'],
      }}
    />
  );
});
